import re

WORD_RE = re.compile(r"[a-zA-Z0-9]+")


def tokenize(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def approx_token_count(text: str) -> int:
    if not text:
        return 0
    return max(1, int(len(text.split()) * 0.75))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0 or not text:
        return ""
    words = text.split()
    approx_tokens = max(1, int(len(words) * 0.75))
    if approx_tokens <= max_tokens:
        return text
    ratio = max_tokens / approx_tokens
    target_words = max(1, int(len(words) * ratio))
    return " ".join(words[:target_words])
