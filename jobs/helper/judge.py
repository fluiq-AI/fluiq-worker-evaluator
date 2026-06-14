import hashlib
import json
import os
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, Optional

from jobs.helper.base import _parse_json_object


JudgeFn = Callable[[str], str]

PROVIDERS = ("openai", "anthropic", "gemini", "fluiq")

DEFAULT_MODELS: Dict[str, str] = {
    "openai":    "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5-20251001",
    "gemini":    "gemini-2.5-flash",
    "fluiq":     "fluiq-judge",
}

_SYSTEM_PROMPT = (
    "You are a strict evaluator. Always respond with a single valid JSON "
    "object and nothing else."
)


class LLMJudge:
    """LLM-as-judge with pluggable providers (openai, anthropic, gemini, fluiq)."""

    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        judge_fn: Optional[JudgeFn] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
    ):
        if provider not in PROVIDERS:
            raise ValueError(
                f"Unsupported judge provider: {provider!r}. Use one of: {PROVIDERS}"
            )
        self.provider = provider
        self.model = model or DEFAULT_MODELS[provider]
        self.temperature = temperature
        self._judge_fn = judge_fn
        self._api_key = api_key
        self._client = None

    def __call__(self, prompt: str) -> str:
        if self._judge_fn is not None:
            return self._judge_fn(prompt)
        if self.provider == "openai":
            return self._call_openai(prompt)
        if self.provider == "anthropic":
            return self._call_anthropic(prompt)
        if self.provider == "gemini":
            return self._call_gemini(prompt)
        if self.provider == "fluiq":
            return self._call_fluiq(prompt)
        raise RuntimeError(f"Unsupported provider: {self.provider}")

    def judge_json(self, prompt: str) -> Dict[str, Any]:
        return _parse_json_object(self(prompt))

    def _call_openai(self, prompt: str) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("judge provider 'openai' requires the `openai` package") from exc
        if self._client is None:
            key = self._api_key or os.getenv("OPENAI_API_KEY")
            self._client = OpenAI(api_key=key) if key else OpenAI()
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content or "{}"

    def _call_anthropic(self, prompt: str) -> str:
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError("judge provider 'anthropic' requires the `anthropic` package") from exc
        if self._client is None:
            key = self._api_key or os.getenv("ANTHROPIC_API_KEY")
            self._client = anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            temperature=self.temperature,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        for block in getattr(resp, "content", []) or []:
            text = getattr(block, "text", None)
            if text:
                return text
        return "{}"

    def _call_gemini(self, prompt: str) -> str:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError("judge provider 'gemini' requires the `google-genai` package") from exc
        if self._client is None:
            key = self._api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            self._client = genai.Client(api_key=key) if key else genai.Client()
        resp = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                temperature=self.temperature,
                response_mime_type="application/json",
            ),
        )
        return getattr(resp, "text", None) or "{}"

    def _call_fluiq(self, prompt: str) -> str:
        import requests
        import config as worker_config
        api_key = self._api_key or os.getenv("FLUIQ_API_KEY")
        if not api_key:
            raise RuntimeError("judge provider 'fluiq' requires FLUIQ_API_KEY env var")
        endpoint = os.getenv("FLUIQ_API_ENDPOINT", "https://api.getfluiq.com/api")
        resp = requests.post(
            f"{endpoint}/v1/judge",
            json={"api_key": api_key, "model": self.model, "prompt": prompt, "temperature": self.temperature},
            timeout=60,
        )
        resp.raise_for_status()
        return (resp.json() or {}).get("content") or "{}"


# ── In-process judge response cache ──────────────────────────────────────────

class InMemoryCache:
    """Thread-safe LRU cache with optional per-entry TTL."""

    def __init__(self, max_size: int = 1000) -> None:
        self._max_size = max_size
        self._cache: OrderedDict = OrderedDict()
        self._expiry: Dict[str, float] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._cache:
                return None
            exp = self._expiry.get(key)
            if exp is not None and time.time() > exp:
                del self._cache[key]
                self._expiry.pop(key, None)
                return None
            self._cache.move_to_end(key)
            return self._cache[key]

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self._max_size:
                    oldest = next(iter(self._cache))
                    del self._cache[oldest]
                    self._expiry.pop(oldest, None)
            self._cache[key] = value
            if ttl is not None:
                self._expiry[key] = time.time() + ttl

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None


class PromptCache:
    """Wraps a judge callable with keyed caching backed by ``InMemoryCache``."""

    def __init__(
        self,
        fn: Callable,
        model: str,
        backend: Optional[InMemoryCache],
        ttl: Optional[float] = None,
    ) -> None:
        self._fn = fn
        self._model = model
        self._backend = backend
        self._ttl = ttl

    def __call__(self, prompt: str, **params: Any) -> str:
        if self._backend is None:
            return self._fn(prompt, **params)
        raw = json.dumps({"model": self._model, "prompt": prompt, **params}, sort_keys=True, default=str)
        key = hashlib.sha256(raw.encode()).hexdigest()
        cached = self._backend.get(key)
        if cached is not None:
            return cached
        result = self._fn(prompt, **params)
        self._backend.set(key, result, ttl=self._ttl)
        return result
