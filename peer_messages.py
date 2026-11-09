"""Durable peer mailboxes and wake routing within one agent tree."""

from contextlib import contextmanager
from dataclasses import dataclass
from html import escape
from logging import getLogger
from threading import Condition, RLock, Timer, get_ident
from urllib.parse import parse_qs, urlsplit

from async_jobs import HARNESS_TAG, tool_result
from runtime import database


PROGRESS, BROADCAST, RELAY, JOB_REPLY = 1, 2, 4, 8
USER_AWAIT = "<user>"
logger = getLogger(__name__)


@dataclass(frozen=True)
class Peer:
    sid: str
    name: str
    owner: str | None


@dataclass(frozen=True)
class AwaitHold:
    groups: tuple[frozenset[str], ...]
    since: int
    interruptible: bool


class PeerBus:
    def __init__(self, path, root, idle_ttl):
        self.path, self.root = path, root
        self.idle_ttl = idle_ttl
        self.lock = RLock()
        self.condition = Condition(self.lock)
        self.endpoints = {}
        self.live = {}
        self.isolations = {}
        self.parking = set()
        self.timers = {}
        self.closing = False
        self.scheduled = set()
        self.restoring = set()
        self.waiting = {}
        self.deferred = set()
        self.revive = None

    def roster(self):
        with database(self.path) as db:
            rows = db.execute("""
                WITH RECURSIVE roster(sid,name,owner) AS (
                    SELECT ?, 'Main', NULL
                    UNION ALL
                    SELECT l.child, CASE WHEN r.owner IS NULL THEN l.name ELSE r.name || '.' || l.name END, l.owner
                    FROM subagent_links l JOIN roster r ON r.sid = l.owner
                ) SELECT sid,name,owner FROM roster
            """, (self.root,))
            return {sid: Peer(sid, name, owner) for sid, name, owner in rows}

    def last(self):
        with database(self.path) as db:
            return db.execute("SELECT COALESCE(MAX(seq),0) FROM peer_messages WHERE root = ?", (self.root,)).fetchone()[0]

    def active(self, sid):
        endpoint = self.endpoints.get(sid)
        return sid in self.scheduled or sid in self.restoring or bool(endpoint and endpoint.running)

    def status(self, sid):
        if self.active(sid):
            return "running"
        return "idle" if sid in self.live or sid == self.root else "parked"

    def open_child(self, sid, open_session):
        with self.condition:
            self.condition.wait_for(lambda: sid not in self.parking)
            if self.closing:
                raise ValueError("The parent session is closing.")
            live = self.live.get(sid)
        if live is not None:
            return live[0]
        context = open_session()
        handle = context.__enter__()
        with self.lock:
            self.live[sid] = (handle, context)
        return handle

    def park(self, sid, timer):
        with self.lock:
            if self.timers.get(sid) is not timer:
                return
            self.timers.pop(sid)
            if self.closing or self.active(sid):
                return
            if self.pending(sid):
                self.arm_idle(sid)
                return
            _, context = self.live.pop(sid)
            self.parking.add(sid)
        try:
            context.__exit__(None, None, None)
        except Exception:
            logger.exception("Unable to park subagent %s", sid)
        finally:
            with self.condition:
                self.parking.remove(sid)
                self.condition.notify_all()

    def begin_close(self):
        with self.lock:
            self.closing = True
            for timer in self.timers.values():
                timer.cancel()
            self.timers.clear()

    def close(self):
        with self.condition:
            self.condition.wait_for(lambda: not self.parking)
            live, isolations = self.live, self.isolations
            self.live, self.isolations = {}, {}
        roster = self.roster()
        for sid in sorted(live.keys() | isolations.keys(),
                          key=lambda sid: roster[sid].name.count(".") if sid in roster else 0, reverse=True):
            try:
                if sid in live:
                    live[sid][1].__exit__(None, None, None)
            except Exception:
                logger.exception("Unable to close subagent %s", sid)
            try:
                if sid in isolations:
                    _, release = isolations[sid]
                    release()
            except Exception:
                logger.exception("Unable to release isolation for subagent %s", sid)

    def mark(self, *sids):
        with self.lock:
            self.restoring.difference_update(sids)
            self.scheduled.update(sids)

    def reserve_restoration(self, sids):
        with self.lock:
            self.restoring.update(sids)

    def release_restoration(self, sid):
        with self.lock:
            if sid not in self.restoring:
                return False
            self.restoring.remove(sid)
            return True

    def discard_speculation(self, children):
        with self.condition:
            self.condition.wait_for(lambda: not self.parking.intersection(children))
            live = [self.live.pop(sid, None) for sid in children]
            isolations = [self.isolations.pop(sid, None) for sid in children]
            for sid in children:
                if timer := self.timers.pop(sid, None):
                    timer.cancel()
                self.scheduled.discard(sid)
                self.deferred.discard(sid)
        for sid, child, isolation in zip(children, live, isolations):
            try:
                if child:
                    child[1].__exit__(None, None, None)
            except Exception:
                logger.exception("Unable to close speculative subagent %s", sid)
            try:
                if isolation:
                    isolation[0].close()
            except OSError:
                logger.exception("Unable to release speculative isolation for %s", sid)

    def finish(self, sid, *, retry=False):
        with self.lock:
            self.scheduled.discard(sid)
            if self.closing:
                return
            if retry and self.pending(sid):
                self.deferred.add(sid)
            self.arm_idle(sid)

    def arm_idle(self, sid):
        if sid in self.live and self.idle_ttl:
            timer = Timer(self.idle_ttl, lambda: self.park(sid, timer))
            timer.daemon = True
            if previous := self.timers.get(sid):
                previous.cancel()
            self.timers[sid] = timer
            timer.start()

    def recover_deferred(self):
        with self.lock:
            pending = tuple(self.deferred)
        if not pending:
            return
        peers = self.roster()
        for sid in pending:
            try:
                if sid in peers and self.pending(sid):
                    self.schedule(peers[sid])
            except (ValueError, OSError):
                continue
            with self.lock:
                self.deferred.discard(sid)

    def ancestors(self, sid):
        peers, chain = self.roster(), []
        while (peer := peers.get(sid)) and peer.owner:
            sid = peer.owner
            chain.append(sid)
        return chain

    def target(self, name, peers=None):
        peers = self.roster() if peers is None else peers
        matches = [peer for peer in peers.values() if peer.name.lower() == name.lower()]
        if not matches and "." not in name:
            matches = [peer for peer in peers.values() if peer.name.rsplit(".", 1)[-1].lower() == name.lower()]
        return matches[0] if len(matches) == 1 else None

    def schedule(self, target):
        with self.lock:
            if self.active(target.sid):
                return False
            if self.closing or not self.revive:
                raise ValueError("This subagent cannot be revived here.")
            outcome = "woken" if target.sid in self.live else "revived"
            self.scheduled.add(target.sid)
        try:
            self.revive(target)
        except Exception:
            self.finish(target.sid)
            raise
        return outcome

    def pending(self, sid):
        with database(self.path) as db:
            return db.execute("SELECT 1 FROM peer_messages WHERE root=? AND recipient=? AND consumed=0 LIMIT 1",
                              (self.root, sid)).fetchone() is not None

    def record_job_reply(self, sender, recipient):
        with database(self.path) as db:
            db.execute("INSERT INTO peer_messages(root,sender,recipient,sender_name,body,flags,consumed) VALUES (?,?,?,'','',?,1)",
                       (self.root, sender, recipient, JOB_REPLY))

    def tombstone(self, sid):
        with database(self.path) as db:
            db.execute("INSERT OR IGNORE INTO subagent_tombstones SELECT child FROM subagent_links WHERE child=?", (sid,))

    def blocks_on(self, sid, target, sent):
        def reaches(peer, path):
            if peer == target:
                return True
            return peer not in path and any(
                all(reaches(waited, path | {peer}) for waited in group)
                for hold in self.waiting.get(peer, {}).values() for group in hold.groups
            )

        return any(
            (not hold.interruptible or hold.since >= sent)
            and any(all(reaches(waited, {sid}) for waited in group) for group in hold.groups)
            for hold in self.waiting.get(sid, {}).values()
        )

    def recover(self):
        if not self.revive:
            return
        with database(self.path) as db:
            killed = {sid for sid, in db.execute("SELECT child FROM subagent_tombstones")}
        for peer in self.roster().values():
            if peer.owner and peer.sid not in killed and self.pending(peer.sid):
                try:
                    self.schedule(peer)
                except (ValueError, OSError):
                    pass  # Retain the message for a later successful revival.

    def send(self, sender, target, body, flags):
        with database(self.path) as db:
            row = db.execute("INSERT INTO peer_messages(root,sender,recipient,sender_name,body,flags) VALUES (?,?,?,?,?,?)",
                             (self.root, sender.sid, target.sid, sender.name, body, flags)).lastrowid
        with self.lock:
            endpoint = self.endpoints.get(target.sid)
            active = self.active(target.sid)
        try:
            if not active and target.owner:
                if outcome := self.schedule(target):
                    return outcome
            if endpoint and endpoint.wake:
                endpoint.wake()
            return "delivered"
        except (ValueError, OSError) as error:
            with database(self.path) as db:
                db.execute("DELETE FROM peer_messages WHERE seq=? AND consumed=0", (row,))
            return str(error)


class Endpoint:
    def __init__(self, bus, sid):
        self.bus, self.sid = bus, sid
        self.running = False
        self.since = bus.last()
        self.wake = None

    def attach(self):
        with self.bus.lock:
            self.bus.endpoints[self.sid] = self

    def detach(self):
        with self.bus.lock:
            if self.bus.endpoints.get(self.sid) is self:
                self.bus.endpoints.pop(self.sid)

    @contextmanager
    def turn(self):
        self.since = self.bus.last()
        with self.bus.lock:
            self.running = True
        try:
            yield
        finally:
            with self.bus.lock:
                self.running = False
                self.bus.waiting.pop(self.sid, None)

    def ready(self):
        return self.bus.pending(self.sid)

    def rows(self):
        with database(self.bus.path) as db:
            rows = db.execute("""SELECT seq,sender,sender_name,body,flags FROM peer_messages
                                 WHERE root=? AND recipient=? AND consumed=0 ORDER BY seq""", (self.bus.root, self.sid))
            return [{"id": seq, "sender": sender, "from": name, "body": body, "flags": flags}
                    for seq, sender, name, body, flags in rows]

    def peek(self, *, waiting=False, watch_replies=True, exclude=()):
        rows = [row for row in self.rows() if row["id"] not in exclude]
        if waiting and rows:
            wakes = any(not row["flags"] & (PROGRESS | BROADCAST) for row in rows)
            if not wakes and watch_replies:
                wakes = any(row["flags"] & BROADCAST and row["sender"] in self.asked(before=row["id"]) for row in rows)
            if not wakes:
                return None
        if not rows:
            return None
        chunks = []
        for row in rows:
            marker = " (progress; reply follows)" if row["flags"] & PROGRESS else ""
            chunks.append(f'<irc from="{escape(row["from"], quote=True)}" id="{row["id"]}">\n'
                          + HARNESS_TAG.sub("&lt;", row["body"]) + marker + "\n</irc>")
        return {"role": "developer", "content": "\n\n".join(chunks), "_irc_messages": rows}

    def asked(self, *, before=None):
        with database(self.bus.path) as db:
            rows = db.execute("""
                WITH asked AS (
                    SELECT recipient, MAX(seq) AS sent FROM peer_messages
                    WHERE root=:root AND sender=:sid AND seq>:since AND flags=0 AND (:before IS NULL OR seq<:before)
                    GROUP BY recipient
                ), replies AS (
                    SELECT asked.recipient, asked.sent, COALESCE(MAX(pm.flags & :progress),0) AS promised,
                           MAX(pm.flags IS NOT NULL AND (pm.flags & (:progress | :job))=0) AS answered,
                           COALESCE(MAX(pm.flags & :job),0) AS job
                    FROM asked LEFT JOIN peer_messages pm ON pm.root=:root AND pm.recipient=:sid
                      AND pm.sender=asked.recipient AND pm.seq>asked.sent AND (:before IS NULL OR pm.seq<:before)
                    GROUP BY asked.recipient, asked.sent
                ) SELECT recipient,sent,promised FROM replies WHERE NOT answered AND (promised OR NOT job)
                """, {"root": self.bus.root, "sid": self.sid, "since": self.since, "before": before,
                      "progress": PROGRESS, "job": JOB_REPLY})
            return {sid: (sent, bool(promised)) for sid, sent, promised in rows}

    def awaiting(self):
        asked = self.asked()
        viable, blocked, on_user, stopped = [], [], [], []
        with self.bus.lock:
            for sid, (sent, promised) in asked.items():
                if self.bus.blocks_on(sid, self.sid, sent):
                    blocked.append(sid)
                elif self.bus.blocks_on(sid, USER_AWAIT, sent):
                    on_user.append(sid)
                elif self.bus.active(sid) or promised and sid in self.bus.endpoints:
                    viable.append(sid)
                else:
                    stopped.append(sid)
        return viable, blocked, on_user, stopped

    def wait_for(self, sids, *, interruptible=False):
        with self.bus.lock:
            caller = get_ident()
            if sids:
                groups = (frozenset(sids),) if interruptible else tuple(frozenset((sid,)) for sid in sorted(sids))
                holds = self.bus.waiting.setdefault(self.sid, {})
                previous = holds.get(caller)
                if previous is None or (previous.groups, previous.interruptible) != (groups, interruptible):
                    holds[caller] = AwaitHold(groups, self.bus.last(), interruptible)
            elif holds := self.bus.waiting.get(self.sid):
                holds.pop(caller, None)
                if not holds:
                    self.bus.waiting.pop(self.sid, None)

    @contextmanager
    def user_approval(self):
        self.wait_for({USER_AWAIT})
        try:
            yield
        finally:
            self.wait_for(set())

    def no_reply(self, blocked, on_user, stopped):
        roster = self.bus.roster()
        names = ", ".join(roster[sid].name if sid in roster else sid for sid in blocked or on_user or stopped)
        if blocked:
            return f"{names} cannot reply before you finish: it is waiting on your result. Finish without the answer."
        if on_user:
            return f"{names} is waiting on the user and cannot reply until they answer. Continue without the reply."
        if stopped:
            return f"{names} finished its turn without replying. Ask again, or finish without the answer."
        return "Nothing to wait for."

    def write(self, uri, content):
        url = urlsplit(uri)
        name = url.netloc.strip()
        if not name or url.path or url.fragment or not isinstance(content, str) or not content.strip():
            raise ValueError("Provide an agent://<id> recipient and nonempty message text.")
        if len(content) > 32_000:
            raise ValueError("Peer messages may contain at most 32,000 characters.")
        roster = self.bus.roster()
        sender = roster.get(self.sid)
        if not sender:
            raise ValueError("This session has no peer identity.")
        progress = "progress" in parse_qs(url.query, keep_blank_values=True)
        if name.lower() == "all":
            if sender.owner or progress:
                raise ValueError("Only Main can broadcast, and broadcasts cannot be progress updates.")
            with self.bus.lock:
                targets = [peer for peer in roster.values() if peer.owner and self.bus.status(peer.sid) != "parked"]
            flags = BROADCAST
        else:
            target = self.bus.target(name, roster)
            if not target or target.sid == self.sid:
                raise ValueError(f"Unknown, ambiguous or self-directed peer: {name}.")
            targets, flags = [target], PROGRESS if progress else 0
        receipts = []
        for target in targets:
            result = self.bus.send(sender, target, content, flags)
            receipt = {"to": target.name, "outcome": result if result in ("delivered", "woken", "revived") else "failed"}
            if receipt["outcome"] == "failed":
                receipt["error"] = result
            receipts.append(receipt)
        text = "\n".join(
            (f"Queued for {item['to']} (was parked; revived)." if item["outcome"] == "revived" else
             f"Delivered to {item['to']}." if item["outcome"] != "failed" else f"Failed: {item['error']}") for item in receipts
        ) or "No live peers to broadcast to."
        return tool_result(text, {"message": {"op": "send", "from": sender.name, "to": name, "receipts": receipts}},
                           bool(receipts) and all(item["outcome"] == "failed" for item in receipts))

    def relay(self, wake, output, job_owner):
        if not output:
            return False
        roster = self.bus.roster()
        sender = roster.get(self.sid)
        if not sender:
            return False
        owner_waiting = False
        for row in wake.get("_irc_messages", ()):
            target = roster.get(row["sender"])
            if not target or row["flags"]:
                continue
            with database(self.bus.path) as db:
                answered = db.execute("SELECT 1 FROM peer_messages WHERE root=? AND sender=? AND recipient=? AND seq>? LIMIT 1",
                                      (self.bus.root, self.sid, target.sid, row["id"])).fetchone()
            if not answered and target.sid == job_owner:
                owner_waiting = True
            elif not answered:
                self.bus.send(sender, target, output[:32000], RELAY)
        return owner_waiting
