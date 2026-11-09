"""Cancellable HTTP transport for the synchronous provider adapters."""

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from concurrent.futures import CancelledError, TimeoutError
from threading import Thread

import httpx


_turn = ContextVar("active_http_turn", default=None)


def active_turn():
    return _turn.get()


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
        future = asyncio.run_coroutine_threadsafe(coroutine, self.loop)
        try:
            if turn:
                with turn.network_call(future):
                    return future.result()
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

    def handle_request(self, request):
        turn = _turn.get()
        if turn:
            turn.check()
        outbound = httpx.Request(
            request.method, request.url, headers=request.headers.raw, content=request.read(),
            extensions=request.extensions.copy(),
        )
        response = self.loop.call(self.client.send(outbound, stream=True), turn)
        return httpx.Response(
            response.status_code, headers=response.headers.raw, stream=ResponseBody(self.loop, response, turn),
            extensions={key: value for key, value in response.extensions.items() if key != "network_stream"},
        )

    def close(self):
        self.loop.call(self.client.aclose())
