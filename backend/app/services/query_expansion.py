import json
import logging
import requests

from app.core.config import Settings

logger = logging.getLogger(__name__)


def expand_query(query: str, settings: Settings, ollama_base_url: str) -> list[str]:
    if not settings.enable_query_expansion:
        return []

    prompt = (
        "Rewrite the user query into up to {count} short alternative searches. "
        "Return a JSON array of strings, no extra text."
    ).format(count=settings.query_expansion_max)

    try:
        response = requests.post(
            f"{ollama_base_url}/api/chat",
            json={
                "model": settings.ollama_model,
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": query},
                ],
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "num_predict": 120,
                }
            },
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        content = result.get("message", {}).get("content", "[]")
        items = json.loads(content)
        if isinstance(items, list):
            cleaned = [item.strip() for item in items if isinstance(item, str)]
            return [item for item in cleaned if item and item.lower() != query.lower()]
    except Exception as exc:
        logger.warning("Query expansion failed: %s", exc)
    return []
