"""Cancellable HTTP transport for the synchronous provider adapters."""

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from concurrent.futures import CancelledError, TimeoutError
from itertools import chain
from threading import Lock, Thread, current_thread

import httpx


_turn = ContextVar("active_http_turn", default=None)


def active_turn():
    return _turn.get()


def sse_frames(lines):
    name, data = None, []
    for line in chain(lines, ("",)):
        if line.startswith("event:"):
            name = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].removeprefix(" "))
        elif not line:
            if data:
                yield name, "\n".join(data)
            name, data = None, []


@contextmanager
def turn_scope(turn):
    token = _turn.set(turn)
    try:
        yield
    finally:
        _turn.reset(token)


def wait_for_result(future):
    turn = _turn.get()
    if not turn:
        return future.result()
    while True:
        turn.check()
        try:
            return future.result(timeout=0.1)
        except TimeoutError:
            if future.done():
                return future.result()


class TransportLoop:
    def __enter__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = Thread(target=self.loop.run_forever, name="mini-pi-http", daemon=True)
        self.thread.start()
        return self

    def call(self, coroutine, turn=None):
        budget = getattr(turn, "network_budget", None)
        if budget is not None:
            try:
                turn.check()
            except BaseException:
                coroutine.close()
                raise
        future = asyncio.run_coroutine_threadsafe(coroutine, self.loop)
        try:
            if turn:
                with turn.network_call(future):
                    if budget is None:
                        return future.result()
                    try:
                        return future.result(timeout=budget())
                    except TimeoutError as error:
                        if not future.done():
                            future.cancel()
                            turn.check()
                            raise turn.deadline_error() from error
                        turn.check()
                        raise
            return future.result()
        except KeyboardInterrupt:
            future.cancel()
            raise
        except CancelledError:
            if turn:
                turn.check()
            raise

    def __exit__(self, *_):
        async def drain():
            pending = set(asyncio.all_tasks()) - {asyncio.current_task()}
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            await self.loop.shutdown_asyncgens()

        try:
            self.call(drain())
        finally:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join()
            self.loop.close()


class ResponseBody(httpx.SyncByteStream):
    def __init__(self, loop, response, turn):
        self.loop, self.response, self.turn = loop, response, turn

    def __iter__(self):
        iterator = self.response.aiter_raw()
        while True:
            try:
                yield self.loop.call(iterator.__anext__(), self.turn)
            except StopAsyncIteration:
                break

    def close(self):
        self.loop.call(self.response.aclose())


class CancellableTransport(httpx.BaseTransport):
    def __init__(self, loop):
        self.loop = loop
        self.client = httpx.AsyncClient()
        self.tail_lock = Lock()
        self.tails = set()

    def drain_tail(self, drain):
        def run():
            try:
                drain()
            finally:
                with self.tail_lock:
                    self.tails.discard(current_thread())

        thread = Thread(target=run, name="mini-pi-http-tail", daemon=True)
        with self.tail_lock:
            self.tails.add(thread)
            thread.start()

    def handle_request(self, request):
        turn = _turn.get()
        if turn:
            turn.check()
            if start := getattr(turn, "start_request", None):
                start()
        outbound = httpx.Request(
            request.method, request.url, headers=request.headers.raw, content=request.read(),
            extensions=request.extensions.copy(),
        )
        response = self.loop.call(self.client.send(outbound, stream=True), turn)
        return httpx.Response(
            response.status_code, headers=response.headers.raw, stream=ResponseBody(self.loop, response, turn),
            extensions={**{key: value for key, value in response.extensions.items() if key != "network_stream"},
                        "pipy_transport": self},
        )

    def close(self):
        with self.tail_lock:
            tails = tuple(self.tails)
        for tail in tails:
            tail.join()
        self.loop.call(self.client.aclose())
