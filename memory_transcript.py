"""Visible conversation shared by the automatic memory backends."""

import re


MEMORY_TAGS = re.compile(r"<(memories|mental_models|hindsight_memories|relevant_memories)>.*?</\1>", re.S)
NON_CONVERSATIONAL = ("_mcp_notification", "_async_jobs", "_service_runs", "_goal_message")


def conversation_text(content):
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(block["text"] for block in content if isinstance(block, dict) and block.get("type") == "text"
                     and isinstance(block.get("text"), str))


def substantive(content):
    return any(char.isalnum() for char in MEMORY_TAGS.sub("", content))


def user_turns(messages):
    return sum(1 for message in messages if message["role"] == "user"
               and not any(key in message for key in NON_CONVERSATIONAL)
               and substantive(conversation_text(message.get("content"))))


def conversation_messages(messages, timestamp=None):
    entries = []
    for message in messages:
        if message["role"] not in ("user", "assistant") or any(key in message for key in NON_CONVERSATIONAL):
            continue
        content = conversation_text(message.get("content"))
        if substantive(content):
            entries.append((message["role"], content, timestamp))
    return entries
