from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Iterator
import json

import requests

from app.core.config import Settings
from app.services.prompts import SYSTEM_PROMPT, build_context
from app.utils.text_utils import approx_token_count, truncate_to_tokens

logger = logging.getLogger(__name__)


@dataclass
class AnswerUsage:
    tokens_in: int | None
    tokens_out: int | None
    cost_estimate: float | None


@dataclass
class AnswerResult:
    answer: str
    model: str | None
    usage: AnswerUsage | None


def estimate_cost(settings: Settings, tokens_in: int | None, tokens_out: int | None) -> float | None:
    # Ollama is free
    return 0.0


def build_messages(
    query: str,
    context_chunks: list[dict],
    chat_history: list[dict] | None,
    settings: Settings,
) -> list[dict]:
    context_text = build_context(context_chunks)
    context_text = truncate_to_tokens(context_text, settings.max_context_tokens)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if chat_history:
        messages.extend(chat_history)
    user_prompt = f"Context:\n{context_text}\n\nQuestion: {query}"
    messages.append({"role": "user", "content": user_prompt})
    return messages


def generate_answer(
    query: str,
    context_chunks: list[dict],
    chat_history: list[dict],
    settings: Settings,
) -> AnswerResult:
    if not context_chunks:
        return AnswerResult(
            answer="No relevant context was retrieved for this question.",
            model=None,
            usage=None,
        )
    
    return _generate_answer_ollama(query, context_chunks, chat_history, settings)


def _generate_answer_ollama(
    query: str,
    context_chunks: list[dict],
    chat_history: list[dict],
    settings: Settings,
) -> AnswerResult:
    context_text = build_context(context_chunks)
    context_text = truncate_to_tokens(context_text, settings.max_context_tokens)
    
    # Build messages for Ollama
    messages = []
    messages.append({"role": "system", "content": SYSTEM_PROMPT})
    
    if chat_history:
        messages.extend(chat_history)
    
    user_prompt = f"Context:\n{context_text}\n\nQuestion: {query}"
    messages.append({"role": "user", "content": user_prompt})
    
    try:
        response = requests.post(
            f"{settings.ollama_base_url}/api/chat",
            json={
                "model": settings.ollama_model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "num_predict": 512,
                }
            },
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        
        content = result.get("message", {}).get("content", "")
        tokens_in = result.get("prompt_eval_count")
        tokens_out = result.get("eval_count")
        cost = 0.0  # Ollama is free
        
        return AnswerResult(
            answer=content,
            model=settings.ollama_model,
            usage=AnswerUsage(tokens_in=tokens_in, tokens_out=tokens_out, cost_estimate=cost),
        )
    except Exception as e:
        logger.error(f"Ollama error: {e}")
        return AnswerResult(
            answer=f"Ollama error: {str(e)}. Make sure Ollama is running on {settings.ollama_base_url}",
            model=None,
            usage=None,
        )


def stream_answer(
    query: str,
    context_chunks: list[dict],
    chat_history: list[dict],
    settings: Settings,
) -> Iterator[str]:
    if not context_chunks:
        yield "No relevant context was retrieved for this question."
        return
    
    yield from _stream_answer_ollama(query, context_chunks, chat_history, settings)


def _stream_answer_ollama(
    query: str,
    context_chunks: list[dict],
    chat_history: list[dict],
    settings: Settings,
) -> Iterator[str]:
    context_text = build_context(context_chunks)
    context_text = truncate_to_tokens(context_text, settings.max_context_tokens)
    
    # Build messages for Ollama
    messages = []
    messages.append({"role": "system", "content": SYSTEM_PROMPT})
    
    if chat_history:
        messages.extend(chat_history)
    
    user_prompt = f"Context:\n{context_text}\n\nQuestion: {query}"
    messages.append({"role": "user", "content": user_prompt})
    
    try:
        response = requests.post(
            f"{settings.ollama_base_url}/api/chat",
            json={
                "model": settings.ollama_model,
                "messages": messages,
                "stream": True,
                "options": {
                    "temperature": 0.2,
                    "num_predict": 512,
                }
            },
            stream=True,
            timeout=60
        )
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                if "message" in chunk and "content" in chunk["message"]:
                    yield chunk["message"]["content"]
    except Exception as e:
        logger.error(f"Ollama streaming error: {e}")
        yield f"Ollama error: {str(e)}. Make sure Ollama is running on {settings.ollama_base_url}"


def estimate_prompt_tokens(query: str, context_chunks: list[dict], settings: Settings) -> int:
    context_text = build_context(context_chunks)
    prompt = f"Context:\n{context_text}\n\nQuestion: {query}"
    return approx_token_count(prompt)
