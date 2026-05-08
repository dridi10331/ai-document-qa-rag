from __future__ import annotations

import json
import logging

from groq import Groq

from app.core.config import Settings

logger = logging.getLogger(__name__)


def expand_query(query: str, settings: Settings) -> list[str]:
    """
    Generate alternative phrasings of the query using Groq LLM.

    Design notes:
    - Query expansion improves recall by covering semantic variants
    - Risk: can drift semantics and hurt precision if over-expanded
    - We cap at settings.query_expansion_max (default 3) to limit noise
    - Results are merged with original query before retrieval
    """
    if not settings.enable_query_expansion:
        return []

    if not settings.groq_api_key:
        logger.warning("Query expansion skipped: GROQ_API_KEY not set")
        return []

    prompt = (
        f"Rewrite the user query into up to {settings.query_expansion_max} short alternative searches. "
        "Return ONLY a JSON array of strings, no explanation, no extra text. "
        "Example: [\"alternative 1\", \"alternative 2\"]"
    )

    try:
        client = Groq(api_key=settings.groq_api_key)
        response = client.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query},
            ],
            temperature=0.3,
            max_tokens=150,
        )
        content = response.choices[0].message.content or "[]"
        # Extract JSON array from response
        start = content.find("[")
        end = content.rfind("]") + 1
        if start >= 0 and end > start:
            content = content[start:end]
        items = json.loads(content)
        if isinstance(items, list):
            cleaned = [item.strip() for item in items if isinstance(item, str) and item.strip()]
            return [item for item in cleaned if item.lower() != query.lower()]
    except Exception as exc:
        logger.warning("Query expansion failed: %s", exc)
    return []
