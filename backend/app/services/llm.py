from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Iterator
import json

import requests
from groq import Groq

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


def get_groq_client(settings: Settings) -> Groq | None:
    if not settings.groq_api_key:
        return None
    return Groq(api_key=settings.groq_api_key)


def estimate_cost(settings: Settings, tokens_in: int | None, tokens_out: int | None) -> float | None:
    # Groq is free tier
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

    if settings.llm_backend == "groq":
        return _generate_answer_groq(query, context_chunks, chat_history, settings)
    elif settings.llm_backend == "ollama":
        return _generate_answer_ollama(query, context_chunks, chat_history, settings)
    else:
        return AnswerResult(
            answer="LLM backend not configured. Set LLM_BACKEND=groq or LLM_BACKEND=ollama.",
            model=None,
            usage=None,
        )


def _generate_answer_groq(
    query: str,
    context_chunks: list[dict],
    chat_history: list[dict],
    settings: Settings,
) -> AnswerResult:
    client = get_groq_client(settings)
    if client is None:
        return AnswerResult(
            answer="LLM service is not configured. Please contact the administrator.",
            model=None,
            usage=None,
        )

    messages = build_messages(query, context_chunks, chat_history, settings)
    try:
        response = client.chat.completions.create(
            model=settings.groq_model,
            messages=messages,
            temperature=0.2,
            max_tokens=512,
        )
        content = response.choices[0].message.content or ""
        tokens_in = response.usage.prompt_tokens if response.usage else None
        tokens_out = response.usage.completion_tokens if response.usage else None
        return AnswerResult(
            answer=content,
            model=settings.groq_model,
            usage=AnswerUsage(tokens_in=tokens_in, tokens_out=tokens_out, cost_estimate=0.0),
        )
    except Exception as e:
        logger.error(f"Groq error: {e}", exc_info=True)
        return AnswerResult(
            answer="An error occurred while generating the answer. Please try again.",
            model=None,
            usage=None,
        )


def _generate_answer_ollama(
    query: str,
    context_chunks: list[dict],
    chat_history: list[dict],
    settings: Settings,
) -> AnswerResult:
    context_text = build_context(context_chunks)
    context_text = truncate_to_tokens(context_text, settings.max_context_tokens)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
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
                "options": {"temperature": 0.2, "num_predict": 512}
            },
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        content = result.get("message", {}).get("content", "")
        tokens_in = result.get("prompt_eval_count")
        tokens_out = result.get("eval_count")
        return AnswerResult(
            answer=content,
            model=settings.ollama_model,
            usage=AnswerUsage(tokens_in=tokens_in, tokens_out=tokens_out, cost_estimate=0.0),
        )
    except Exception as e:
        logger.error(f"Ollama error: {e}", exc_info=True)
        return AnswerResult(
            answer="An error occurred while generating the answer. Please try again.",
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

    if settings.llm_backend == "groq":
        yield from _stream_answer_groq(query, context_chunks, chat_history, settings)
    elif settings.llm_backend == "ollama":
        yield from _stream_answer_ollama(query, context_chunks, chat_history, settings)
    else:
        yield "LLM backend not configured. Set LLM_BACKEND=groq or LLM_BACKEND=ollama."


def _stream_answer_groq(
    query: str,
    context_chunks: list[dict],
    chat_history: list[dict],
    settings: Settings,
) -> Iterator[str]:
    client = get_groq_client(settings)
    if client is None:
        yield "LLM service is not configured. Please contact the administrator."
        return

    messages = build_messages(query, context_chunks, chat_history, settings)
    try:
        stream = client.chat.completions.create(
            model=settings.groq_model,
            messages=messages,
            temperature=0.2,
            max_tokens=512,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content
    except Exception as e:
        logger.error(f"Groq streaming error: {e}", exc_info=True)
        yield "An error occurred while streaming the answer. Please try again."


def _stream_answer_ollama(
    query: str,
    context_chunks: list[dict],
    chat_history: list[dict],
    settings: Settings,
) -> Iterator[str]:
    context_text = build_context(context_chunks)
    context_text = truncate_to_tokens(context_text, settings.max_context_tokens)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
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
                "options": {"temperature": 0.2, "num_predict": 512}
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
        logger.error(f"Ollama streaming error: {e}", exc_info=True)
        yield "An error occurred while streaming the answer. Please try again."


def estimate_prompt_tokens(query: str, context_chunks: list[dict], settings: Settings) -> int:
    context_text = build_context(context_chunks)
    prompt = f"Context:\n{context_text}\n\nQuestion: {query}"
    return approx_token_count(prompt)
