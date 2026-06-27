import re


def redact_sensitive_data(text: str) -> str:
    text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[REDACTED_EMAIL]", text)
    text = re.sub(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\d{10}|\d{3}[-.\s]\d{3}[-.\s]\d{4})\b", "[REDACTED_PHONE]", text)
    return text


def refuses_system_prompt_leak(text: str) -> bool:
    normalized = text.lower()
    return "system prompt" in normalized or "developer message" in normalized
