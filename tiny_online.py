"""Resolve lightweight background models through the normal authenticated router."""

from collections import deque

from auth_broker import AuthBrokerError
from model_profile import route_base


LOCAL = object()


def online_candidates(profile, roles, credential, *, current=None, fallback=True):
    """Collect role primaries, their retry chains, then the current model's chain."""
    if not profile:
        return (current,) if current else ()
    seen, ordered, queue = set(), [], deque()

    def add(route):
        identity = route_base(route), route.endpoint(), route.api
        if identity in seen:
            return False
        seen.add(identity)
        ordered.append(route)
        return True

    def primary(role):
        for selector, effort in profile.expand("@" + role):
            pattern = selector if effort is None else f"{selector}:{effort}"
            names = profile.choices(pattern)
            if not names and selector.startswith("local/"):
                return LOCAL
            for name in names:
                route = profile.routes[name]
                try:
                    if credential(route) is not None:
                        return route
                except AuthBrokerError:
                    continue
        return None

    for role in roles:
        route = primary(role)
        if route is LOCAL:
            # An explicit local primary must not turn into a billable request.
            if not ordered:
                return ()
            continue
        if route is None:
            continue
        add(route)
        if not fallback:
            return tuple(ordered)
        queue.append((route, role))

    def expand():
        visited = set()
        while queue:
            route, hint = queue.popleft()
            effort = route.effort(route.reasoning_effort)
            key = profile.chain_key(route, effort, hint)
            if not key or (key, route.name) in visited:
                continue
            visited.add((key, route.name))
            for name in profile.candidates(key, route, effort):
                candidate = profile.routes[name]
                if add(candidate):
                    queue.append((candidate, None))

    if fallback:
        expand()
    if current and (fallback or not ordered):
        if add(current) and fallback:
            queue.append((current, None))
            expand()
    return tuple(ordered)
