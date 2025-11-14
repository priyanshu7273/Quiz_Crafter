"""LLM provider orchestration and graceful fallback handling."""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Protocol

from app.config import get_settings
from app.utils.logger import configure_logger, log_extra

logger = configure_logger(__name__)


@dataclass
class LLMResult:
    """Normalised response returned by every provider."""

    provider: str
    model: str
    content: str
    latency: float
    usage: Dict[str, Any]


class LLMProvider(Protocol):
    """Runtime protocol implemented by each backend provider."""

    name: str

    async def generate(
        self,
        *,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
        timeout: int,
    ) -> LLMResult:
        ...

    def healthy(self) -> bool:
        ...


def _import_optional(module: str) -> Any:
    try:  # pragma: no cover - optional dependency support
        return __import__(module, fromlist=["*"])
    except ModuleNotFoundError:
        return None


class OpenAIProvider:
    """Wrapper around the OpenAI Chat Completions API."""

    name = "openai"

    def __init__(self, api_key: str, organization: Optional[str], models: Iterable[str]) -> None:
        openai = _import_optional("openai")
        if openai is None:
            raise RuntimeError("openai package not installed")
        self._client = openai.AsyncOpenAI(api_key=api_key, organization=organization)
        self._models = list(models)
        self._cursor = 0

    def healthy(self) -> bool:  # pragma: no cover - simple property
        return True

    async def generate(
        self,
        *,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
        timeout: int,
    ) -> LLMResult:
        model = self._models[self._cursor % len(self._models)]
        self._cursor += 1

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        start = time.perf_counter()
        response = await self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        latency = time.perf_counter() - start
        choice = response.choices[0].message.content
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
        return LLMResult("openai", model, choice or "", latency, usage)


class GeminiProvider:
    name = "gemini"

    def __init__(self, api_key: str, models: Iterable[str]) -> None:
        genai = _import_optional("google.generativeai")
        if genai is None:
            raise RuntimeError("google-generativeai package not installed")
        genai.configure(api_key=api_key)
        self._genai = genai
        self._models = list(models)
        self._cursor = 0

    def healthy(self) -> bool:  # pragma: no cover
        return True

    async def generate(
        self,
        *,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
        timeout: int,
    ) -> LLMResult:
        model_name = self._models[self._cursor % len(self._models)]
        self._cursor += 1
        model = self._genai.GenerativeModel(model_name)
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        start = time.perf_counter()
        response = await asyncio.to_thread(
            model.generate_content,
            full_prompt,
            generation_config=self._genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
            safety_settings=None,
        )
        latency = time.perf_counter() - start
        usage = {
            "prompt_tokens": getattr(response.usage_metadata, "prompt_token_count", 0),
            "completion_tokens": getattr(response.usage_metadata, "candidates_token_count", 0),
            "total_tokens": getattr(response.usage_metadata, "total_token_count", 0),
        }
        return LLMResult("gemini", model_name, response.text or "", latency, usage)


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, api_key: str, models: Iterable[str]) -> None:
        anthropic = _import_optional("anthropic")
        if anthropic is None:
            raise RuntimeError("anthropic package not installed")
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._models = list(models)
        self._cursor = 0

    def healthy(self) -> bool:  # pragma: no cover
        return True

    async def generate(
        self,
        *,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
        timeout: int,
    ) -> LLMResult:
        model = self._models[self._cursor % len(self._models)]
        self._cursor += 1
        start = time.perf_counter()
        response = await self._client.messages.create(
            model=model,
            system=system_prompt or "",
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        latency = time.perf_counter() - start
        first_block = response.content[0]
        content = getattr(first_block, "text", "") if hasattr(first_block, "text") else str(first_block)
        usage = {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
        }
        return LLMResult("anthropic", model, content, latency, usage)


class PerplexityProvider:
    name = "perplexity"

    def __init__(self, api_key: str, models: Iterable[str]) -> None:
        aiohttp = _import_optional("aiohttp")
        if aiohttp is None:
            raise RuntimeError("aiohttp package not installed")
        self._aiohttp = aiohttp
        self._api_key = api_key
        self._models = list(models)
        self._cursor = 0

    def healthy(self) -> bool:  # pragma: no cover
        return True

    async def generate(
        self,
        *,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
        timeout: int,
    ) -> LLMResult:
        model = self._models[self._cursor % len(self._models)]
        self._cursor += 1
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        start = time.perf_counter()
        async with self._aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.perplexity.ai/chat/completions",
                headers=headers,
                json=payload,
                timeout=timeout,
            ) as response:
                data = await response.json()
        latency = time.perf_counter() - start
        choice = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = data.get("usage", {})
        return LLMResult("perplexity", model, choice, latency, usage)


class LLMManager:
    """Entry point used by the rest of the application."""

    def __init__(self) -> None:
        settings = get_settings()
        self._providers: List[LLMProvider] = []
        if settings.openai_api_key:
            try:
                self._providers.append(
                    OpenAIProvider(settings.openai_api_key, settings.openai_organization, settings.openai_models)
                )
            except Exception as exc:  # pragma: no cover - optional dependency guard
                logger.warning(
                    "Unable to configure OpenAI provider",
                    extra=log_extra(error=str(exc)),
                )
        if settings.anthropic_api_key:
            try:
                self._providers.append(AnthropicProvider(settings.anthropic_api_key, settings.anthropic_models))
            except Exception as exc:
                logger.warning(
                    "Unable to configure Anthropic provider",
                    extra=log_extra(error=str(exc)),
                )
        if settings.gemini_api_key:
            try:
                self._providers.append(GeminiProvider(settings.gemini_api_key, settings.gemini_models))
            except Exception as exc:
                logger.warning(
                    "Unable to configure Gemini provider",
                    extra=log_extra(error=str(exc)),
                )
        if settings.perplexity_api_key:
            try:
                self._providers.append(PerplexityProvider(settings.perplexity_api_key, settings.perplexity_models))
            except Exception as exc:
                logger.warning(
                    "Unable to configure Perplexity provider",
                    extra=log_extra(error=str(exc)),
                )
        if not self._providers:
            raise RuntimeError("No LLM providers available – supply at least one API key")

    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> LLMResult:
        """Call each configured provider until one succeeds."""

        settings = get_settings()
        last_error: Optional[Exception] = None
        for provider in list(self._providers):
            if not provider.healthy():  # pragma: no cover - currently always true
                continue
            try:
                logger.info(
                    "Dispatching prompt to provider",
                    extra=log_extra(provider=provider.name),
                )
                return await provider.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=settings.request_timeout_seconds,
                )
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Provider failed, falling back",
                    extra=log_extra(provider=provider.name, error=str(exc)),
                )
        raise RuntimeError("All configured LLM providers failed") from last_error

    def stats(self) -> Dict[str, Any]:
        return {provider.name: {"healthy": provider.healthy()} for provider in self._providers}
