"""Anthropic server-side refusal fallback wire policy."""

from model_catalog import lookup, token_prices
from observed_usage import anthropic_counters


BETA = "server-side-fallback-2026-06-01"
THINKING = {"thinking", "redacted_thinking"}
DECLINED = THINKING | {"tool_use"}

def validate_chain(value, model, route):
    if not isinstance(value, list) or len(value) > 3:
        raise ValueError(f"Route {route!r} requires up to three Anthropic fallback entries.")
    seen = {model}
    for entry in value:
        if not isinstance(entry, dict) or entry.keys() - {"model", "max_tokens", "thinking", "output_config", "speed"}:
            raise ValueError(f"Route {route!r} has an invalid Anthropic fallback entry.")
        target = entry.get("model")
        limit = entry.get("max_tokens")
        if (
            not isinstance(target, str) or not target or target != target.strip() or target in seen
            or "max_tokens" in entry and (type(limit) is not int or limit <= 0)
            or "speed" in entry and entry["speed"] != "fast"
            or any(key in entry and not isinstance(entry[key], dict) for key in ("thinking", "output_config"))
        ):
            raise ValueError(f"Route {route!r} has an invalid or duplicate Anthropic fallback model.")
        seen.add(target)
    return tuple(value)


def chain_betas(chain):
    if not chain:
        return ()
    betas = [BETA]
    if any(entry.get("speed") == "fast" for entry in chain):
        betas.append("fast-mode-2026-02-01")
    if any(entry.get("output_config", {}).get("effort") for entry in chain):
        betas.append("effort-2025-11-24")
    if any(entry.get("output_config", {}).get("task_budget") for entry in chain):
        betas.append("task-budgets-2026-03-13")
    return tuple(betas)


def marker(block):
    source, target = block.get("from"), block.get("to")
    if isinstance(source, dict) and isinstance(target, dict):
        before, after = source.get("model"), target.get("model")
        if all(isinstance(model, str) and model.strip() for model in (before, after)):
            return {"type": "fallback", "from": {"model": before}, "to": {"model": after}}
    return None


def retained(blocks):
    boundary = max((index for index, block in blocks.items() if block["type"] == "fallback"), default=-1)
    return {
        index: block for index, block in blocks.items()
        if block["type"] != "_ignored_fallback" and (index >= boundary or block["type"] not in DECLINED)
    }


def collapse_segments(blocks):
    segment = reasoning_segment = 0
    for block in blocks:
        if block["type"] == "fallback":
            segment += 1
        elif block["type"] in THINKING:
            reasoning_segment = segment
    if not segment:
        return blocks
    output, segment = [], 0
    for block in blocks:
        if block["type"] == "fallback":
            segment += 1
        elif block["type"] not in THINKING or segment == reasoning_segment:
            output.append(block)
    return output


def tool_calls_last(blocks):
    saw_tool = False
    for block in blocks:
        if block["type"] == "tool_use":
            saw_tool = True
        elif saw_tool and block["type"] not in THINKING:
            return sorted(blocks, key=lambda item: item["type"] == "tool_use")
    return blocks


def iterations(raw):
    value = raw.get("iterations")
    return tuple(item for item in value if isinstance(item, dict)) if isinstance(value, list) else ()


def served_model(raw):
    return next((
        model for item in reversed(iterations(raw))
        if item.get("type") == "fallback_message" and isinstance(model := item.get("model"), str) and model.strip()
    ), None)


def iteration_cost(raw, model, prices, first_party):
    attempts = iterations(raw)
    if not attempts:
        return None
    requested_model = lookup("anthropic", model) if first_party else None
    has_fallback = any(item.get("type") == "fallback_message" for item in attempts)
    amounts = [0, 0, 0, 0]
    for item in attempts:
        prompt, output, read, written, _ = anthropic_counters(item, "5m")
        fallback = item.get("type") == "fallback_message"
        if has_fallback and not fallback and not (output or written):
            continue
        inputs = max(0, prompt - read - written)
        if fallback:
            read += inputs
            inputs = 0
        reported = item.get("model")
        reported = reported.strip() if isinstance(reported, str) else None
        served = lookup("anthropic", reported) if first_party and reported and reported != model else None
        rates = token_prices(served, prompt) or prices or token_prices(requested_model, prompt)
        if rates is None:
            return None
        amounts[0] += inputs * rates.input
        amounts[1] += output * rates.output
        amounts[2] += read * rates.read
        amounts[3] += written * rates.write_5m
    return dict(zip(("input", "output", "cacheRead", "cacheWrite", "total"),
                    (amount / 1_000_000 for amount in (*amounts, sum(amounts)))))
