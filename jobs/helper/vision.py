"""Vision-grounded evaluation — judge an answer against the image(s) it was about.

The SDK stores a payload-free ``_media_ref`` for each media part (see
``fluiq-sdk .../shared/media.py``). This module:

  * :func:`extract_media` — pull usable image media out of a trace event /
    messages / explicit inputs. A ref is *usable* only when it carries a URL
    (``source="url"``) — base64 media is not stored in the trace, so it can only
    be judged when the caller passes it inline (``data``).
  * ``build_*_content`` — assemble provider-specific multimodal message content
    (text prompt + images) for the vision judge.
  * :class:`VisionFaithfulness` — score whether the answer faithfully/accurately
    describes the attached image(s), using the multimodal judge.

Pure helpers (extraction + content building) are unit-tested; the judge call is
stubbed via ``LLMJudge._multimodal_fn`` in tests.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from jobs.helper import judge_prompts
from jobs.helper.base import BaseEvaluator, EvalResult, _clamp_unit, judge_score
from jobs.helper.judge import LLMJudge

_MAX_IMAGES = 6  # cap images per judge call


# ── media extraction ─────────────────────────────────────────────────────────

_IMAGE_ONLY = ("image",)
_ALL_MEDIA = ("image", "audio", "video")


def _kind_from_mime(mime: Any) -> Optional[str]:
    if not isinstance(mime, str) or "/" not in mime:
        return None
    top = mime.split("/", 1)[0].lower()
    return top if top in _ALL_MEDIA else None


def _media_from_ref(ref: Dict[str, Any], kinds) -> Optional[Dict[str, Any]]:
    """A stored ``_media_ref`` → a judge media item, or None if unusable."""
    kind = ref.get("kind") or _kind_from_mime(ref.get("mime"))
    if kind not in kinds:
        return None
    # Inline base64 payload (kept for small media) is directly usable.
    if ref.get("data"):
        return {"kind": kind, "mime": ref.get("mime"), "data": ref["data"]}
    if ref.get("source") == "url" and ref.get("url"):
        return {"kind": kind, "mime": ref.get("mime"), "url": ref["url"]}
    # base64 ref without inline data carries only a hash → not usable.
    return None


def _media_from_raw_part(part: Dict[str, Any], kinds) -> Optional[Dict[str, Any]]:
    """A raw (un-stripped) content part → a judge media item, or None.

    Covers OpenInference-ingested traces and inline eval inputs that still carry
    the actual url / base64 (OpenAI ``image_url`` / ``input_audio``, Anthropic
    ``source``, Gemini ``inline_data`` / ``file_data``)."""
    iu = part.get("image_url")
    if "image" in kinds:
        if isinstance(iu, dict) and iu.get("url"):
            return {"kind": "image", "url": iu["url"]}
        if isinstance(iu, str) and iu:
            return {"kind": "image", "url": iu}
    # OpenAI audio part: {"input_audio": {"data", "format"}}
    for k in ("input_audio", "audio"):
        av = part.get(k)
        if isinstance(av, dict) and av.get("data") and "audio" in kinds:
            fmt = av.get("format")
            return {"kind": "audio", "mime": f"audio/{fmt}" if fmt else None, "data": av["data"]}
    src = part.get("source")
    if isinstance(src, dict):
        mime = src.get("media_type")
        kind = _kind_from_mime(mime) or "image"
        if kind in kinds:
            if src.get("url"):
                return {"kind": kind, "mime": mime, "url": src["url"]}
            if src.get("data"):
                return {"kind": kind, "mime": mime, "data": src["data"]}
    idata = part.get("inline_data") or part.get("inlineData")
    if isinstance(idata, dict) and idata.get("data"):
        mime = idata.get("mime_type") or idata.get("mimeType")
        kind = _kind_from_mime(mime) or "image"
        if kind in kinds:
            return {"kind": kind, "mime": mime, "data": idata["data"]}
    fdata = part.get("file_data") or part.get("fileData")
    if isinstance(fdata, dict) and (fdata.get("file_uri") or fdata.get("fileUri")):
        mime = fdata.get("mime_type") or fdata.get("mimeType")
        kind = _kind_from_mime(mime) or "image"
        if kind in kinds:
            return {"kind": kind, "mime": mime, "url": fdata.get("file_uri") or fdata.get("fileUri")}
    return None


def _walk_content(content: Any, out: List[Dict[str, Any]], kinds) -> None:
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            ref = part.get("_media_ref")
            if isinstance(ref, dict):
                item = _media_from_ref(ref, kinds)
            else:
                item = _media_from_raw_part(part, kinds)
            if item:
                out.append(item)


def extract_media(
    *,
    messages: Any = None,
    media: Any = None,
    event: Optional[Dict[str, Any]] = None,
    kinds=_IMAGE_ONLY,
) -> List[Dict[str, Any]]:
    """Collect usable media of the requested ``kinds`` from any of: explicit
    ``media`` (inline items), ``messages`` (content parts), or a trace ``event``.
    Deduped, capped."""
    out: List[Dict[str, Any]] = []

    if isinstance(media, list):
        for m in media:
            if not isinstance(m, dict) or not (m.get("url") or m.get("data")):
                continue
            kind = m.get("kind") or _kind_from_mime(m.get("mime")) or "image"
            if kind not in kinds:
                continue
            out.append({"kind": kind, "mime": m.get("mime"),
                        **({"url": m["url"]} if m.get("url") else {"data": m["data"]})})

    if event and messages is None:
        messages = event.get("messages") or event.get("contents") or event.get("input")
    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict):
                _walk_content(msg.get("content"), out, kinds)
            elif isinstance(msg, list):
                _walk_content(msg, out, kinds)

    # dedupe by (url|data-prefix), cap
    seen: set = set()
    unique: List[Dict[str, Any]] = []
    for m in out:
        key = m.get("url") or (m.get("data") or "")[:64]
        if key and key not in seen:
            seen.add(key)
            unique.append(m)
    return unique[:_MAX_IMAGES]


# ── provider content builders ────────────────────────────────────────────────

def _data_uri(item: Dict[str, Any]) -> str:
    mime = item.get("mime") or "image/png"
    return f"data:{mime};base64,{item['data']}"


def build_openai_content(prompt: str, media: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for m in media:
        kind = m.get("kind", "image")
        if kind == "audio" and m.get("data"):
            fmt = (m.get("mime") or "audio/wav").split("/", 1)[-1]
            content.append({"type": "input_audio", "input_audio": {"data": m["data"], "format": fmt}})
            continue
        if kind == "image":
            url = m.get("url") or (_data_uri(m) if m.get("data") else None)
            if url:
                content.append({"type": "image_url", "image_url": {"url": url}})
        # OpenAI chat has no video content type — video is skipped here.
    return content


def build_anthropic_content(prompt: str, media: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for m in media:
        # Anthropic supports images (and documents); audio/video are skipped.
        if m.get("kind", "image") != "image":
            continue
        if m.get("data"):
            content.append({"type": "image", "source": {
                "type": "base64", "media_type": m.get("mime") or "image/png", "data": m["data"]}})
        elif m.get("url"):
            content.append({"type": "image", "source": {"type": "url", "url": m["url"]}})
    return content


def build_gemini_parts(prompt: str, media: List[Dict[str, Any]]) -> List[Any]:
    # Gemini natively handles image, audio AND video via mime type.
    parts: List[Any] = [prompt]
    for m in media:
        default_mime = {"audio": "audio/wav", "video": "video/mp4"}.get(m.get("kind"), "image/png")
        if m.get("data"):
            parts.append({"inline_data": {"mime_type": m.get("mime") or default_mime, "data": m["data"]}})
        elif m.get("url"):
            parts.append({"file_data": {"mime_type": m.get("mime") or default_mime, "file_uri": m["url"]}})
    return parts


# ── provider media capability ────────────────────────────────────────────────
# Which media kinds each judge provider can accept in a multimodal call.
_PROVIDER_MEDIA = {
    "openai":    {"image", "audio"},   # gpt-4o vision + audio
    "anthropic": {"image"},            # images (and documents)
    "gemini":    {"image", "audio", "video"},  # natively multimodal
}


def supported_kinds(provider: str, requested) -> tuple:
    allowed = _PROVIDER_MEDIA.get(provider, set())
    return tuple(k for k in requested if k in allowed)


# ── evaluator ────────────────────────────────────────────────────────────────

class _MediaGroundedBase(BaseEvaluator):
    """Shared: does the answer faithfully describe the attached media?"""

    prompt_name = "vision_faithfulness"
    kinds: tuple = _IMAGE_ONLY

    def __init__(self, judge: Optional[LLMJudge] = None, threshold: float = 0.7):
        super().__init__(threshold=threshold)
        self.judge = judge or LLMJudge()

    def evaluate(self, answer: str, question: Optional[str] = None,
                 media: Any = None, messages: Any = None, event: Optional[Dict] = None,
                 **_: Any) -> EvalResult:
        # Only request kinds this provider can actually judge.
        kinds = supported_kinds(self.judge.provider, self.kinds)
        if not kinds:
            return self._result(1.0, f"judge provider {self.judge.provider!r} has no media path",
                                {"applicable": False, "media": 0})
        items = extract_media(media=media, messages=messages, event=event, kinds=kinds)
        if not items:
            # No usable media (none present, or only hash-only base64 refs) →
            # not applicable, neutral pass.
            return self._result(1.0, "no media available to evaluate",
                                {"applicable": False, "media": 0})
        if not answer or not answer.strip():
            return self._result(0.0, "empty answer", {"applicable": True, "media": len(items)})

        by_kind: Dict[str, int] = {}
        for m in items:
            by_kind[m["kind"]] = by_kind.get(m["kind"], 0) + 1

        prompt = judge_prompts.render(
            self.prompt_name,
            question=(question or "(no question captured)"),
            answer=answer,
        )
        data = self.judge.judge_multimodal_json(prompt, items)
        score = judge_score(data)
        return self._result(
            score, str(data.get("reason") or ""),
            {"applicable": True, "media": len(items), "by_kind": by_kind,
             "unsupported_claims": data.get("unsupported_claims") or []},
        )


class VisionFaithfulness(_MediaGroundedBase):
    """Image-only grounding (back-compat metric name)."""
    name = "vision_faithfulness"
    prompt_name = "vision_faithfulness"
    kinds = _IMAGE_ONLY


class MediaFaithfulness(_MediaGroundedBase):
    """Grounding against all attached media — image, audio, and video."""
    name = "media_faithfulness"
    prompt_name = "media_faithfulness"
    kinds = _ALL_MEDIA
