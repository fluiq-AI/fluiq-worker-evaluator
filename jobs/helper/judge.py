import hashlib
import json
import os
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, Optional

from jobs.helper import judge_prompts
from jobs.helper.base import _parse_json_object


JudgeFn = Callable[[str], str]

# Every provider polygate can route a judge call to. Kept in step with
# fluiq-api/shared/providers.py — the two deployables can't share code, and a
# provider a customer can store a key for but the judge refuses to route is the
# failure this list exists to prevent.
PROVIDERS = (
    "openai", "anthropic", "gemini", "mistral", "groq", "together",
    "fireworks", "perplexity", "xai", "cerebras", "deepseek", "moonshot",
    "zai",
    # Gateways. Fixed host, OpenAI wire format, someone else's models behind
    # it — so a model id here is usually "vendor/model".
    "openrouter", "vercel", "baseten", "deepinfra", "sambanova",
    "nebius", "novita", "hyperbolic",
    # The clouds (bedrock/azure/vertex/databricks/cloudflare) are deliberately
    # absent: polygate can reach them, but each needs an endpoint or region
    # stored beside the key, which provider keys cannot yet carry. Kept in step
    # with ROUTABLE_PROVIDERS in fluiq-api/shared/providers.py.
)

# Every judge provider is routed through polygate (Fluiq's own unified LLM
# client) rather than each vendor's SDK, so the worker speaks one request/
# response shape and gains key rotation + backoff for free. The vision path
# stays on native SDKs — polygate's unified string-content messages can't
# express image blocks across all three providers (Gemini in particular).
_POLYGATE_TEXT_PROVIDERS = PROVIDERS

# Providers whose chat API accepts ``response_format``. Anthropic has no JSON
# mode on this path and relies on the system prompt plus ``_parse_json_object``.
_JSON_MODE_PROVIDERS = frozenset({
    "openai", "moonshot", "groq", "together", "fireworks",
    "mistral", "xai", "cerebras", "deepseek", "zai",
    # Gateways inherit whatever the upstream model supports. JSON mode is
    # near-universal on the models these serve, and a provider that ignores the
    # field returns prose that _parse_json_object still recovers.
    "openrouter", "vercel", "baseten", "deepinfra", "sambanova",
    "nebius", "novita", "hyperbolic",
})

# The narrower set that honours strict ``json_schema`` with an enum — real
# constrained decoding. The others get plain JSON mode; a schema they reject
# would fail the call outright, which is worse than a slightly looser answer the
# caller validates anyway.
_JSON_SCHEMA_PROVIDERS = frozenset({"openai", "moonshot", "fireworks", "xai"})

# Judge calls are idempotent, so transient provider failures (429 / 5xx /
# Anthropic's 529 overload) are safe to retry. polygate rotates across the key
# pool first and only sleeps once every key is busy. Configurable; set to 1 to
# disable and fall back to a single attempt per call.
_JUDGE_RETRY_ATTEMPTS = max(1, int(os.getenv("EVAL_JUDGE_RETRY_ATTEMPTS", "3") or 3))


def _judge_retry():
    """A polygate ``Retry`` policy, or ``None`` when retries are disabled."""
    if _JUDGE_RETRY_ATTEMPTS <= 1:
        return None
    from polygate import Retry
    return Retry(max_attempts=_JUDGE_RETRY_ATTEMPTS)


class JudgeUsage:
    """Judge tokens actually spent, accumulated across every call that hit a provider.

    Evals are the compute-heavy part of the product and the only part whose cost
    scales with trace size, jury size, and judge model — none of which the
    per-metric row count can see. Without this, judge spend is unmeasurable:
    a 3-model jury over a 40-step trajectory and a one-shot relevance check are
    indistinguishable after the fact.

    A single instance is shared by the primary judge and every juror for one
    eval message, so a panel's cost lands in one place. Cache hits deliberately
    do not accumulate — a served-from-cache verdict spends no tokens, and
    billing should reflect that.
    """

    __slots__ = ("input_tokens", "output_tokens", "calls", "_lock")

    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0
        self._lock = threading.Lock()

    def add(self, input_tokens: int, output_tokens: int) -> None:
        with self._lock:
            self.input_tokens += max(0, int(input_tokens or 0))
            self.output_tokens += max(0, int(output_tokens or 0))
            self.calls += 1

    def drain(self) -> tuple[int, int, int]:
        """Return ``(input, output, calls)`` and reset to zero.

        One eval message can persist several metric rows. Draining means the
        totals land on the first row and later rows carry zeros, so a SUM over
        rows is the true per-message spend rather than a multiple of it.
        """
        with self._lock:
            totals = (self.input_tokens, self.output_tokens, self.calls)
            self.input_tokens = self.output_tokens = self.calls = 0
            return totals

# The model used when a provider is chosen without one. Cheap-and-fast per
# provider: a judge runs on every scored trace, so the default has to be the one
# nobody regrets leaving in place.
DEFAULT_MODELS: Dict[str, str] = {
    "openai":     "gpt-4o-mini",
    "anthropic":  "claude-haiku-4-5-20251001",
    "gemini":     "gemini-2.5-flash",
    "mistral":    "mistral-small-latest",
    "groq":       "llama-3.3-70b-versatile",
    "xai":        "grok-3-mini",
    "deepseek":   "deepseek-chat",
    "moonshot":   "kimi-k2-0711-preview",
}

def _usage_openai(resp: Any) -> tuple[int, int]:
    u = getattr(resp, "usage", None)
    if u is None:
        return (0, 0)
    return (getattr(u, "prompt_tokens", 0) or 0, getattr(u, "completion_tokens", 0) or 0)


def _usage_anthropic(resp: Any) -> tuple[int, int]:
    u = getattr(resp, "usage", None)
    if u is None:
        return (0, 0)
    # Cache reads/writes are billed differently but are still input tokens; count
    # them so the total reflects what the provider actually charged for.
    cached = (getattr(u, "cache_read_input_tokens", 0) or 0) + (
        getattr(u, "cache_creation_input_tokens", 0) or 0
    )
    return ((getattr(u, "input_tokens", 0) or 0) + cached, getattr(u, "output_tokens", 0) or 0)


def _usage_gemini(resp: Any) -> tuple[int, int]:
    u = getattr(resp, "usage_metadata", None)
    if u is None:
        return (0, 0)
    return (
        getattr(u, "prompt_token_count", 0) or 0,
        getattr(u, "candidates_token_count", 0) or 0,
    )


# ── Forced-choice response schemas ───────────────────────────────────────────
#
# The shape both schemas describe: {"choice": <one of the labels>, "reason": str}.
# Providers that support constrained decoding get the enum, which makes an
# off-menu answer impossible rather than merely unlikely.

def _openai_choice_schema(labels: list[str]) -> Dict[str, Any]:
    """OpenAI/Moonshot strict JSON-schema mode."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "judge_choice",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "choice": {"type": "string", "enum": list(labels)},
                    "reason": {"type": "string"},
                },
                "required": ["choice", "reason"],
                "additionalProperties": False,
            },
        },
    }


def _gemini_choice_schema(labels: list[str]) -> Dict[str, Any]:
    """Gemini response schema (OpenAPI subset — no additionalProperties)."""
    return {
        "type": "OBJECT",
        "properties": {
            "choice": {"type": "STRING", "enum": list(labels)},
            "reason": {"type": "STRING"},
        },
        "required": ["choice", "reason"],
    }


class LLMJudge:
    """LLM-as-judge with pluggable providers (openai, anthropic, gemini, moonshot)."""

    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        judge_fn: Optional[JudgeFn] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        usage: Optional["JudgeUsage"] = None,
    ):
        if provider not in PROVIDERS:
            raise ValueError(
                f"Unsupported judge provider: {provider!r}. Use one of: {PROVIDERS}"
            )
        self.provider = provider
        resolved = model or DEFAULT_MODELS.get(provider)
        if not resolved:
            # Some providers host so many third-party models that no default is
            # defensible (Together, Fireworks, Cerebras). Saying so beats a
            # KeyError, and beats picking one on the customer's behalf.
            raise ValueError(
                f"Judge provider {provider!r} has no default model; pass one "
                f"explicitly (e.g. judge='{provider}:<model>')."
            )
        self.model = resolved
        self.temperature = temperature
        self._judge_fn = judge_fn
        # Optional test/override hook for the multimodal path: (prompt, media) -> str.
        # Kept separate from ``_judge_fn`` so the text-only judge cache never
        # swallows the images.
        self._multimodal_fn: Optional[Callable[[str, list], str]] = None
        self._api_key = api_key
        self._client = None
        # Shared across the primary judge and its jurors when a panel is built,
        # so one message's spend accumulates in a single place.
        self.usage = usage if usage is not None else JudgeUsage()

    def _record(self, input_tokens: Any, output_tokens: Any) -> None:
        """Record one provider call. Never raises — usage is telemetry, not the answer.

        Provider SDKs differ in where they hang usage and occasionally omit it
        (streaming, cached responses, older API versions). A missing count must
        under-report rather than fail the evaluation the customer asked for.
        """
        try:
            self.usage.add(input_tokens, output_tokens)
        except Exception:  # noqa: BLE001
            pass

    def __call__(self, prompt: str) -> str:
        if self._judge_fn is not None:
            return self._judge_fn(prompt)
        if self.provider in _POLYGATE_TEXT_PROVIDERS:
            return self._chat_via_polygate(prompt)
        raise RuntimeError(f"Unsupported provider: {self.provider}")

    def judge_json(self, prompt: str) -> Dict[str, Any]:
        return _parse_json_object(self(prompt))

    def judge_choice(self, prompt: str, labels: list[str]) -> Dict[str, Any]:
        """Make the judge pick one of ``labels`` rather than invent a number.

        Choosing between written options is a classification task a model does
        reliably; emitting a calibrated float is not, and free floats cluster
        around whatever threshold the author set. The caller maps the chosen
        label to a score through its own table, so the number never comes from
        the model at all.

        Where the provider can constrain decoding to an enum (OpenAI/Moonshot
        JSON-schema mode, Gemini response schemas) it is constrained; elsewhere
        the prompt states the options and the caller validates the answer against
        them. Both paths return ``{"choice": str, "reason": str}`` — a choice
        outside the set is left as-is for the caller to reject, because silently
        coercing it would invent a grade.
        """
        if self._judge_fn is not None:
            # The wrapper chain (cache, BYOK error reporting) forwards kwargs to
            # the provider call, so the enum survives it. A test hook that takes
            # only a prompt still works: the options are in the prompt text, and
            # the caller validates the answer either way.
            try:
                return _parse_json_object(self._judge_fn(prompt, choice_labels=labels))
            except TypeError:
                return _parse_json_object(self._judge_fn(prompt))
        return _parse_json_object(self._chat_via_polygate(prompt, choice_labels=labels))

    # ── multimodal (vision) judging ──────────────────────────────────────────
    def supports_vision(self) -> bool:
        """Whether this provider path can attach images to the judge call."""
        return self.provider in ("openai", "anthropic", "gemini")

    def judge_multimodal_json(self, prompt: str, media: list) -> Dict[str, Any]:
        """Judge ``prompt`` with image ``media`` attached; returns parsed JSON.

        ``media`` items are normalized dicts: ``{kind, mime, url?|data?}``.
        The multimodal path deliberately bypasses the text-only judge cache
        (``_judge_fn``) so images are never dropped."""
        if self._multimodal_fn is not None:
            return _parse_json_object(self._multimodal_fn(prompt, media))
        if self.provider == "openai":
            return _parse_json_object(self._call_openai_mm(prompt, media))
        if self.provider == "anthropic":
            return _parse_json_object(self._call_anthropic_mm(prompt, media))
        if self.provider == "gemini":
            return _parse_json_object(self._call_gemini_mm(prompt, media))
        raise RuntimeError(f"provider {self.provider!r} does not support vision judging")

    def _call_openai_mm(self, prompt: str, media: list) -> str:
        from jobs.helper.vision import build_openai_content
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
                {"role": "system", "content": judge_prompts.system_prompt()},
                {"role": "user", "content": build_openai_content(prompt, media)},
            ],
            response_format={"type": "json_object"},
        )
        self._record(*_usage_openai(resp))
        return resp.choices[0].message.content or "{}"

    def _call_anthropic_mm(self, prompt: str, media: list) -> str:
        from jobs.helper.vision import build_anthropic_content
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
            system=judge_prompts.system_prompt(),
            messages=[{"role": "user", "content": build_anthropic_content(prompt, media)}],
        )
        self._record(*_usage_anthropic(resp))
        for block in getattr(resp, "content", []) or []:
            text = getattr(block, "text", None)
            if text:
                return text
        return "{}"

    def _call_gemini_mm(self, prompt: str, media: list) -> str:
        from jobs.helper.vision import build_gemini_parts
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
            contents=build_gemini_parts(prompt, media),
            config=types.GenerateContentConfig(
                system_instruction=judge_prompts.system_prompt(),
                temperature=self.temperature,
                response_mime_type="application/json",
            ),
        )
        self._record(*_usage_gemini(resp))
        return getattr(resp, "text", None) or "{}"

    # ── text judging via polygate ─────────────────────────────────────────────
    # These keep their per-provider names because run.py's BYOK wrapper calls
    # them directly, but each just defers to the unified polygate path — the
    # provider is already fixed on the instance, so there's nothing to branch on.
    # Per-provider seams. They all reach the same transport, but they are the
    # documented place to stub a provider, so the dispatch keeps going through
    # them. ``choice_labels`` is keyword-only with a default so a stub that takes
    # only a prompt still satisfies the signature.

    def _call_openai(self, prompt: str, *, choice_labels: Optional[list] = None) -> str:
        return self._chat_via_polygate(prompt, choice_labels=choice_labels)

    def _call_moonshot(self, prompt: str, *, choice_labels: Optional[list] = None) -> str:
        return self._chat_via_polygate(prompt, choice_labels=choice_labels)

    def _call_anthropic(self, prompt: str, *, choice_labels: Optional[list] = None) -> str:
        return self._chat_via_polygate(prompt, choice_labels=choice_labels)

    def _call_gemini(self, prompt: str, *, choice_labels: Optional[list] = None) -> str:
        return self._chat_via_polygate(prompt, choice_labels=choice_labels)

    def _chat_via_polygate(
        self, prompt: str, choice_labels: Optional[list[str]] = None,
    ) -> str:
        """One request/response shape for every text provider, via polygate.

        The system prompt goes in as a ``system`` message; polygate maps it to
        each provider's native shape (Anthropic's top-level ``system`` field,
        Gemini's ``systemInstruction``, an OpenAI/Moonshot system message).
        JSON mode is requested where the provider supports it; Anthropic has no
        JSON mode, so it relies on the system prompt plus ``_parse_json_object``
        downstream, exactly as the SDK path did.

        ``choice_labels`` additionally pins the answer to an enum where the
        provider can enforce one, so the model cannot return a label that was
        never offered.
        """
        try:
            from polygate import chat as polygate_chat
        except ImportError as exc:  # pragma: no cover - packaging guard
            raise RuntimeError(
                "judge text providers require the `polygate` package"
            ) from exc

        provider = self.provider
        messages = [
            {"role": "system", "content": judge_prompts.system_prompt()},
            {"role": "user",   "content": prompt},
        ]
        extra: Dict[str, Any] = {}
        temperature: Optional[float] = self.temperature
        max_tokens: Optional[int] = None

        # Pass the BYOK key when we have one, else None so polygate reads the
        # provider's own env var. For Moonshot that var is MOONSHOT_API_KEY —
        # never OPENAI_API_KEY — so a keyless Moonshot judge refuses rather than
        # borrowing an OpenAI credential, the guard the old SDK path enforced by
        # hand. Gemini additionally honours GOOGLE_API_KEY, which polygate does
        # not read, so resolve that fallback here.
        api_key = self._api_key
        if not api_key and provider == "gemini":
            api_key = os.getenv("GOOGLE_API_KEY")

        if provider in _JSON_MODE_PROVIDERS:
            # Strict json_schema is only honoured by a subset; the rest accept
            # json_object and are held to the option set by the prompt plus the
            # caller's validation. Sending a schema they ignore is harmless, but
            # sending one they *reject* would fail the whole call, so the two
            # tiers are kept apart.
            if choice_labels and provider in _JSON_SCHEMA_PROVIDERS:
                extra["response_format"] = _openai_choice_schema(choice_labels)
            else:
                extra["response_format"] = {"type": "json_object"}
        elif provider == "gemini":
            # polygate overwrites the whole generationConfig with this kwarg, so
            # it has to carry the temperature too; a separate temperature= would
            # be discarded.
            extra["generationConfig"] = {
                "temperature": self.temperature,
                "responseMimeType": "application/json",
            }
            if choice_labels:
                extra["generationConfig"]["responseSchema"] = _gemini_choice_schema(
                    choice_labels
                )
            temperature = None
        elif provider == "anthropic":
            max_tokens = 1024  # Anthropic requires an explicit max_tokens.
            # Anthropic has no JSON/enum mode on this path, so its answer is
            # constrained by the prompt and validated by the caller instead.

        resp = polygate_chat(
            provider=provider,
            model=self.model,
            messages=messages,
            api_key=api_key or None,
            temperature=temperature,
            max_tokens=max_tokens,
            retry=_judge_retry(),
            **extra,
        )
        usage = getattr(resp, "usage", None)
        if usage is not None:
            self._record(usage.prompt_tokens, usage.completion_tokens)
        return resp.content or "{}"


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
    """Wraps a judge callable with keyed caching backed by ``InMemoryCache``.

    ``namespace`` is the tenancy boundary. The backing cache is a single
    process-wide instance shared by every eval the worker handles, so the key
    MUST carry whatever distinguishes one caller's judge call from another's —
    otherwise two orgs whose rendered judge prompts collide share a verdict,
    leaking one org's judged content to the other. Callers pass the
    organization id; when the credential paying for the call stops being
    global (per-org provider keys), append its fingerprint here too.
    """

    def __init__(
        self,
        fn: Callable,
        model: str,
        backend: Optional[InMemoryCache],
        ttl: Optional[float] = None,
        namespace: str = "",
    ) -> None:
        self._fn = fn
        self._model = model
        self._backend = backend
        self._ttl = ttl
        self._namespace = namespace

    def __call__(self, prompt: str, **params: Any) -> str:
        if self._backend is None:
            return self._fn(prompt, **params)
        # ``params`` is nested rather than splatted so a param named "model" or
        # "prompt" can't shadow the fields above it and collapse distinct calls
        # onto one key.
        raw = json.dumps(
            {
                "namespace": self._namespace,
                "model": self._model,
                "prompt": prompt,
                "params": params,
            },
            sort_keys=True,
            default=str,
        )
        key = hashlib.sha256(raw.encode()).hexdigest()
        cached = self._backend.get(key)
        if cached is not None:
            return cached
        result = self._fn(prompt, **params)
        self._backend.set(key, result, ttl=self._ttl)
        return result
