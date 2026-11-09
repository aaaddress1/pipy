"""Execute saved reset plans through the same fenced path as manual redemption."""

import asyncio
import logging
import sqlite3
import sys
from concurrent.futures import Future
from contextlib import closing, nullcontext
from dataclasses import dataclass, replace
from queue import Empty, SimpleQueue

from auth_broker import AuthBroker, AuthBrokerError, broker_settings, request_scopes
from auto_reset_plan import (
    CLAUDE, CODEX, COOLDOWN, DEFER, PlanState, ResetPolicy, claude_cached_statuses, codex_live_reports,
    headless_actions, plan_claude, plan_codex,
)
from broker_resets import BrokerResets
from broker_usage import usage_reports
from quota_reset import ResetService
from quota_service import now_ms
from reset_expiry import claim_expiry_notice, terminal_text


logger = logging.getLogger(__name__)
PREFIX = {CLAUDE: "claudeResets", CODEX: "codexResets"}
SETTLED = {"reset", "already_redeemed", "no_credit", "spent"}


@dataclass(frozen=True)
class ResetConfirmation:
    question: str
    answer: Future
    foreground: bool = False


class AutoReset:
    def __init__(self, pool, state, *, interactive=False):
        self.pool = pool
        self.state = state / "sessions.sqlite3"
        settings = broker_settings(pool.profile)
        self.policies = {provider: ResetPolicy.from_settings(settings, prefix) for provider, prefix in PREFIX.items()}
        self.attempted, self.deferred, self.last = set(), {}, {}
        self.inflight, self.notified, self.adopted = {}, set(), {}
        self.waiters = {}
        self.interactive = interactive
        self.sweeps, self.last_sweep = {}, {}
        self.confirmations = SimpleQueue()
        self.foreground_confirmations = SimpleQueue()
        self.idle_answers = {}
        self.monitor = None
        self.remote = isinstance(pool, AuthBroker)
        self.broker_resets = BrokerResets(pool) if self.remote else None
        self.service = None

    def consent(self, provider, mode=None):
        with closing(sqlite3.connect(self.state, timeout=2)) as db, db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS reset_consent(
                    profile TEXT NOT NULL, provider TEXT NOT NULL, mode TEXT NOT NULL, PRIMARY KEY(profile,provider)
                )
            """)
            fields = self.pool.profile, provider
            if mode:
                db.execute("INSERT OR REPLACE INTO reset_consent(profile,provider,mode) VALUES(?,?,?)", (*fields, mode))
                return mode
            row = db.execute("SELECT mode FROM reset_consent WHERE profile=? AND provider=?", fields).fetchone()
            return row[0] if row else "unset"

    async def policy(self, provider):
        policy = self.policies[provider]
        if policy.mode == "unset":
            return replace(policy, mode=await asyncio.to_thread(self.consent, provider))
        return policy

    def local(self):
        if self.service is None:
            self.service = ResetService(self.pool.database, self.pool.quota, self.pool.oauth, self.pool.refresh_locks)
        return self.service

    async def statuses(self, provider, active):
        async with asyncio.timeout(10):
            if self.remote:
                rows = await self.broker_resets.list(provider, evidence=True, active=active)
            else:
                rows = await self.local().list(provider, evidence=True, active=active)
        allowed, _ = self.pool.pool.get(provider, self.pool.pool.get("*", (None, True)))
        return [row for row in rows if allowed is None or row.get("identityKey") in allowed]

    async def reports(self, provider, *, force=False):
        if self.remote:
            if force:
                await self.pool.request("POST", "/v1/usage/stale", params={"provider": provider})
            response = await self.pool.request("GET", "/v1/usage", params={"provider": provider})
            raw = self.pool.payload(response).get("reports")
        else:
            if force:
                await self.pool.quota.invalidate(provider)
            entries = await self.pool.sqlite(self.pool.database.entries, (provider,))
            raw = await self.pool.quota.reports([entry for entry in entries if self.pool.entry(entry)], provider)
        return usage_reports({"reports": raw}, provider)

    def permits(self, route, identifier):
        row = self.pool.rows.get(identifier)
        if not row or row["provider"] != route.broker_provider or row["kind"] != "oauth":
            return False
        if route.broker_accounts is not None and row["identity"] not in route.broker_accounts:
            return False
        scopes = request_scopes(route.broker_provider, route.quota_scope(), route.model)
        return not any(scope in scopes and (scope in ("auth", "account-policy") or scope.startswith("model-policy:"))
                       for _, scope in self.pool.active_blocks(row, self.pool.server_time()))

    async def revalidate(self, provider, identifier):
        await self.pool.invalidate_reset(provider, identifier)
        if self.remote:
            await self.pool.fetch()
        else:
            await self.pool.ensure()

    async def approved(self, policy, provider, actions, ask, notice):
        if policy.mode == "no":
            return []
        if not actions or policy.mode == "yes":
            return actions
        if ask is not None:
            answer = await ask(provider, actions)
            if answer in ("yes", "no"):
                await asyncio.to_thread(self.consent, provider, answer)
            return actions if answer == "yes" else []
        approved = headless_actions(actions)
        waiting = next((action for action in actions if action not in approved and action.attempt_key not in self.notified), None)
        if waiting:
            self.notified.add(waiting.attempt_key)
            notice(f"Saved resets are eligible for {provider}. Run /usage reset or set {PREFIX[provider]}.autoRedeem.")
        for action in approved:
            notice(f"Spending a saved {provider} reset before it expires for {action.label}.")
        return approved

    async def execute(self, provider, actions, statuses, route, notice):
        restored = []
        for action in actions:
            identifier = action.target["credentialId"]
            status = statuses.get(identifier)
            current = now_ms()
            if not status or action.attempt_key in self.attempted or current - self.last.get(action.account_key, 0) < COOLDOWN:
                continue
            if action.reason == "blocked-account" and not self.permits(route, identifier):
                continue
            credit_id = action.target.get("creditId") or action.credit_id
            self.attempted.add(action.attempt_key)
            self.last[action.account_key] = current
            try:
                service = self.broker_resets if self.remote else self.local()
                outcome = await service.redeem(provider, identifier, credit_id, action.available_count,
                                               status.get("identityKey"))
            except (AuthBrokerError, TimeoutError, sqlite3.Error, OSError):
                outcome = {"code": "network_error"}
            code = outcome.get("code")
            stamp = outcome.get("attemptedAt")
            adopted = code == "reset_recent" and isinstance(stamp, int) and stamp > self.adopted.get((provider, identifier), 0)
            if code == "reset" or adopted:
                self.adopted[provider, identifier] = stamp if isinstance(stamp, int) else current
                try:
                    await self.revalidate(provider, identifier)
                except (AuthBrokerError, sqlite3.Error):
                    pass
                restored.append(identifier)
                if code == "reset":
                    notice(f"Spent a saved {provider} reset for {action.label}.")
            if code not in SETTLED and not adopted:
                self.attempted.discard(action.attempt_key)
                self.deferred[action.attempt_key] = now_ms() + DEFER
                if action.reason == "blocked-account":
                    notice(f"Saved {provider} reset returned {code or 'unknown'} for {action.label}.")
        return restored

    async def blocked(self, route, active, hint, ask, notice):
        provider = route.broker_provider
        if provider not in PREFIX or route.key_override:
            return False, None
        try:
            policy = await self.policy(provider)
            if policy.mode == "no":
                return False, None
            await self.pool.ensure()
            account = self.pool.rows.get(active)
            if not account or account["provider"] != provider or account["kind"] != "oauth":
                return False, None
            if ask is not None and (idle_answer := self.idle_answers.get(provider)):
                idle_answer.cancel()
            key = (provider, active, route.model, route.quota_scope(), route.broker_accounts)
            task = self.inflight.get(key)
            if task is None or task.done():
                task = asyncio.create_task(self.plan_blocked(route, active, account["account"], hint, policy, ask, notice))
                self.inflight[key] = task

                def finished(done):
                    if self.inflight.get(key) is done:
                        self.inflight.pop(key, None)
                    if not done.cancelled():
                        done.exception()

                task.add_done_callback(finished)
            self.waiters[task] = self.waiters.get(task, 0) + 1
            try:
                return await asyncio.shield(task)
            finally:
                self.waiters[task] -= 1
                if not self.waiters[task]:
                    self.waiters.pop(task)
                    if not task.done():
                        task.cancel()
        except (AuthBrokerError, TimeoutError, sqlite3.Error, OSError, ValueError):
            logger.debug("Saved-reset eligibility check failed.", exc_info=True)
            return False, None

    async def plan_blocked(self, route, active, identity, hint, policy, ask, notice):
        provider = route.broker_provider
        reports = await self.reports(provider, force=True) if provider == CODEX else []
        statuses = await self.statuses(provider, active)
        state = PlanState(now_ms(), policy, self.attempted, self.deferred, self.last,
                          lambda identifier: self.permits(route, identifier))
        if provider == CODEX:
            reports = codex_live_reports(reports, statuses, state.now, synthesize_active=True)
            actions, _ = plan_codex(state, reports, blocked=True, model=route.model, identity=identity, hint=hint)
        else:
            actions, _ = plan_claude(state, statuses, blocked=True, model=route.model, hint=hint)
        approved = [] if self.interactive and ask is None and policy.mode == "unset" else (
            await self.approved(policy, provider, actions, ask, notice)
        )
        if approved:
            restored = await self.execute(provider, approved, {row["credentialId"]: row for row in statuses}, route, notice)
            return any(self.permits(route, identifier) for identifier in restored), None
        if not actions:
            for row in statuses:
                identifier, stamp = row["credentialId"], row.get("recentResetAt")
                if (isinstance(stamp, int) and stamp > self.adopted.get((provider, identifier), 0)
                    and self.permits(route, identifier)):
                    self.adopted[provider, identifier] = stamp
                    await self.revalidate(provider, identifier)
                    return True, None
        retries = [row["retryAfterMs"] / 1000 for row in statuses
                   if row.get("error") and "retryAfterMs" in row and self.permits(route, row["credentialId"])]
        return False, min(retries, default=None) if not actions and provider == CLAUDE and policy.mode == "yes" else None

    def schedule(self, provider, reports):
        current = now_ms()
        if (not reports or provider not in PREFIX or provider in self.sweeps
            or current - self.last_sweep.get(provider, 0) < COOLDOWN):
            return
        if not self.interactive and (self.inflight or self.policies[provider].mode == "no"):
            return
        self.last_sweep[provider] = current
        self.sweeps[provider] = task = asyncio.create_task(self.sweep(provider, reports))

        def finished(done):
            self.sweeps.pop(provider, None)
            if not done.cancelled() and done.exception():
                logger.debug("Saved-reset sweep failed.", exc_info=done.exception())

        task.add_done_callback(finished)

    async def sweep(self, provider, reports):
        if self.interactive:
            path = self.state if self.remote else self.pool.database.path
            try:
                message = await asyncio.to_thread(claim_expiry_notice, path, reports, now_ms())
                if message:
                    terminal_notice(message)
            except sqlite3.Error:
                logger.debug("Saved-reset expiry notice could not be recorded.", exc_info=True)
        if self.inflight:
            return
        policy = await self.policy(provider)
        if policy.mode == "no":
            return
        state = PlanState(now_ms(), policy, self.attempted, self.deferred, self.last)
        if provider == CLAUDE:
            await self.pool.ensure()
            accounts = [{"credentialId": identifier, **row["account"]} for identifier, row in self.pool.rows.items()
                        if row["provider"] == provider and row["kind"] == "oauth"]
            candidates, _ = plan_claude(state, claude_cached_statuses(accounts, reports), reports)
            if not candidates:
                return
        statuses = await self.statuses(provider, None)
        state = replace(state, now=now_ms())
        actions = self.salvage_plan(provider, state, reports, statuses)
        ask = self.ask_idle if self.interactive else None
        approved = await self.approved(policy, provider, actions, ask, terminal_notice)
        if approved and self.interactive and policy.mode == "unset":
            # The user may have taken time to answer. Re-plan, retaining only the grants they saw.
            offered = {(action.attempt_key, action.credit_id): action.available_count for action in approved}
            reports = await self.reports(provider) if provider == CODEX else reports
            statuses = await self.statuses(provider, None)
            state = replace(state, now=now_ms(), policy=await self.policy(provider))
            approved = [action for action in self.salvage_plan(provider, state, reports, statuses)
                        if action.available_count >= offered.get((action.attempt_key, action.credit_id), float("inf"))]
        await self.execute(provider, approved, {row["credentialId"]: row for row in statuses}, None, terminal_notice)

    @staticmethod
    def salvage_plan(provider, state, reports, statuses):
        if provider == CLAUDE:
            return plan_claude(state, statuses, reports)[0]
        return plan_codex(state, codex_live_reports(reports, statuses, state.now))[0]

    async def ask_idle(self, provider, actions):
        return await self.ask_confirmation(provider, actions, foreground=False)

    async def ask_foreground(self, provider, actions):
        return await self.ask_confirmation(provider, actions, foreground=True)

    async def ask_confirmation(self, provider, actions, *, foreground):
        answer = Future()
        if not foreground:
            self.idle_answers[provider] = answer
        queue = self.foreground_confirmations if foreground else self.confirmations
        queue.put(ResetConfirmation(confirmation_text(provider, actions), answer, foreground))
        try:
            remaining = min(action.expires_in for action in actions) / 1000
            async with asyncio.timeout(min(300, remaining)):
                return await asyncio.wrap_future(answer)
        except TimeoutError:
            return None
        finally:
            if self.idle_answers.get(provider) is answer:
                self.idle_answers.pop(provider, None)
            answer.cancel()

    def next_confirmation(self, *, foreground_only=False):
        queues = (self.foreground_confirmations,) if foreground_only else (self.foreground_confirmations, self.confirmations)
        for queue in queues:
            while True:
                try:
                    prompt = queue.get_nowait()
                except Empty:
                    break
                if not prompt.answer.done():
                    return prompt

    async def start_idle(self):
        if self.monitor is not None:
            return

        async def poll():
            while True:
                for provider in self.pool.model_providers & PREFIX.keys():
                    try:
                        self.schedule(provider, await self.pool.usage(provider))
                    except (AuthBrokerError, OSError, sqlite3.Error):
                        logger.debug("Idle quota refresh failed.", exc_info=True)
                await asyncio.sleep(300)

        self.monitor = asyncio.create_task(poll())

    async def close(self):
        if self.monitor:
            self.monitor.cancel()
            await asyncio.gather(self.monitor, return_exceptions=True)
        tasks = (*self.inflight.values(), *self.sweeps.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if self.service:
            await self.service.close()
        if self.broker_resets:
            await self.broker_resets.close()


def terminal_notice(message):
    print(f"[{terminal_text(message)}]", file=sys.stderr)


def with_terminal_consent(transport, operation, turn):
    """Keep console input on its own thread while the transport loop handles I/O."""
    requests = SimpleQueue()

    async def ask(provider, actions):
        answer = asyncio.get_running_loop().create_future()
        requests.put((provider, actions, answer))
        return await answer

    def resolve(answer, value):
        if not answer.done():
            answer.set_result(value)

    result = asyncio.run_coroutine_threadsafe(operation(ask), transport.loop)
    try:
        with turn.network_call(result) if turn else nullcontext():
            while not result.done():
                try:
                    provider, actions, answer = requests.get(timeout=.1)
                except Empty:
                    continue
                if not result.done():
                    value = confirm_on_terminal(provider, actions)
                    transport.loop.call_soon_threadsafe(resolve, answer, value)
            return result.result()
    except BaseException:
        result.cancel()
        raise


def confirmation_text(provider, actions):
    label = "Claude" if provider == CLAUDE else "Codex"
    lines = [f"Spend {len(actions)} saved {label} reset(s)?"]
    for action in actions:
        if action.remaining is not None:
            detail = f"{' + '.join(action.blocked_windows)} blocked for {action.remaining / 60_000:.0f} min"
        else:
            detail = f"credit expires in {action.expires_in / 60_000:.0f} min"
            if action.salvage_used is not None:
                detail += f", {action.salvage_window} is {action.salvage_used:.0%} used"
        title = f" ({terminal_text(action.title)})" if action.title else ""
        lines.append(f"  {terminal_text(action.label)}{title}. {detail}")
    return "\n".join(lines)


CONSENT_PROMPT = "Enter yes or no to remember the policy, or press Enter to dismiss: "


def consent_value(value):
    answer = value.strip().lower()
    return {"y": "yes", "n": "no", "yes": "yes", "no": "no"}.get(answer)


def confirm_on_terminal(provider, actions):
    print(confirmation_text(provider, actions), flush=True)
    try:
        return consent_value(input(CONSENT_PROMPT))
    except (EOFError, KeyboardInterrupt):
        return None
