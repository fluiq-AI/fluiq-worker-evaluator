"""Vision-grounded evaluation — media extraction, content builders, evaluator.

Judge call stubbed via LLMJudge._multimodal_fn (no network). Run:
    ../.workers-venv/Scripts/python.exe tests/test_vision.py
"""
import json
import os
import sys

os.environ.setdefault("EVAL_JUDGE_THRESHOLD", "0.7")
os.environ.setdefault("EVAL_JUDGE_CACHE_TTL", "300")
os.environ.setdefault("EVAL_JUDGE_CACHE_MAX", "1000")
os.environ.setdefault("CLICKHOUSE_PORT", "8123")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jobs.helper.judge import LLMJudge
from jobs.helper.vision import (
    MediaFaithfulness, VisionFaithfulness, build_anthropic_content, build_gemini_parts,
    build_openai_content, extract_media, supported_kinds,
)

# A stored media ref with a URL (usable) vs base64 (payload not stored → unusable).
URL_REF = {"type": "image_url", "_media_ref": {"kind": "image", "mime": "image/png", "source": "url", "url": "https://x/cat.png", "sha256": "abc"}}
B64_REF = {"type": "image_url", "_media_ref": {"kind": "image", "mime": "image/png", "source": "base64", "bytes": 900, "sha256": "def"}}


def _vision_judge(score=0.4, claims=None):
    payload = json.dumps({"score": score, "unsupported_claims": claims or ["a purple hat"], "reason": "stub"})
    j = LLMJudge(provider="openai")
    j._multimodal_fn = lambda prompt, media: payload
    return j


def test_extract_media_url_ref_usable():
    msgs = [{"role": "user", "content": [{"type": "text", "text": "what is this?"}, URL_REF]}]
    media = extract_media(messages=msgs)
    assert len(media) == 1 and media[0]["url"] == "https://x/cat.png"


def test_extract_media_base64_ref_unusable():
    # base64 ref carries no payload → cannot be judged from the trace.
    msgs = [{"role": "user", "content": [B64_REF]}]
    assert extract_media(messages=msgs) == []


def test_extract_media_inline_and_raw():
    # explicit inline media + a raw (un-stripped) OpenAI image_url part
    inline = [{"kind": "image", "mime": "image/jpeg", "data": "QkFTRTY0"}]
    msgs = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://y/dog.png"}}]}]
    media = extract_media(media=inline, messages=msgs)
    urls_datas = {m.get("url") or m.get("data") for m in media}
    assert "https://y/dog.png" in urls_datas and "QkFTRTY0" in urls_datas


def test_content_builders():
    media = [{"kind": "image", "url": "https://x/a.png"}, {"kind": "image", "mime": "image/jpeg", "data": "QUJD"}]
    o = build_openai_content("grade this", media)
    assert o[0] == {"type": "text", "text": "grade this"}
    assert o[1]["image_url"]["url"] == "https://x/a.png"
    assert o[2]["image_url"]["url"].startswith("data:image/jpeg;base64,QUJD")

    a = build_anthropic_content("p", media)
    assert a[1]["source"]["type"] == "url"
    assert a[2]["source"]["type"] == "base64" and a[2]["source"]["data"] == "QUJD"

    g = build_gemini_parts("p", media)
    assert g[0] == "p"
    assert g[1]["file_data"]["file_uri"] == "https://x/a.png"
    assert g[2]["inline_data"]["data"] == "QUJD"


def test_vision_faithfulness_scores_with_media():
    ev = VisionFaithfulness(judge=_vision_judge(score=0.4), threshold=0.7)
    msgs = [{"role": "user", "content": [{"type": "text", "text": "Describe the image"}, URL_REF]}]
    res = ev.evaluate(answer="A dog wearing a purple hat.", question="Describe the image", messages=msgs)
    assert abs(res.score - 0.4) < 1e-6 and res.passed is False
    assert res.details["applicable"] is True and res.details["media"] == 1
    assert res.details["unsupported_claims"] == ["a purple hat"]


def test_vision_faithfulness_no_media_not_applicable():
    ev = VisionFaithfulness(judge=_vision_judge(), threshold=0.7)
    res = ev.evaluate(answer="Some text answer", question="q", messages=[{"role": "user", "content": "no image"}])
    assert res.score == 1.0 and res.passed is True
    assert res.details["applicable"] is False and res.details["media"] == 0


def test_vision_faithfulness_from_event():
    ev = VisionFaithfulness(judge=_vision_judge(score=0.9), threshold=0.7)
    event = {"messages": [{"role": "user", "content": [URL_REF]}]}
    res = ev.evaluate(answer="an answer", event=event)
    assert res.score == 0.9 and res.details["media"] == 1


# ── audio / video (MediaFaithfulness) ────────────────────────────────────────

def test_media_faithfulness_audio_gemini():
    # Gemini supports audio; inline base64 audio ref is usable.
    audio_ref = {"type": "input_audio", "_media_ref": {"kind": "audio", "mime": "audio/wav", "source": "base64", "data": "QUJD", "sha256": "h"}}
    judge = LLMJudge(provider="gemini")
    judge._multimodal_fn = lambda p, m: json.dumps({"score": 0.6, "reason": "transcript ok"})
    ev = MediaFaithfulness(judge=judge, threshold=0.7)
    res = ev.evaluate(answer="The speaker says hello.", messages=[{"role": "user", "content": [audio_ref]}])
    assert abs(res.score - 0.6) < 1e-6
    assert res.details["media"] == 1 and res.details["by_kind"] == {"audio": 1}


def test_media_faithfulness_audio_skipped_on_anthropic():
    # Anthropic has no audio path → audio filtered out → not applicable.
    audio = [{"kind": "audio", "mime": "audio/wav", "data": "QUJD"}]
    judge = LLMJudge(provider="anthropic")
    judge._multimodal_fn = lambda p, m: json.dumps({"score": 0.5})
    ev = MediaFaithfulness(judge=judge, threshold=0.7)
    res = ev.evaluate(answer="x", media=audio)
    assert res.details["applicable"] is False and res.details["media"] == 0


def test_supported_kinds():
    assert supported_kinds("gemini", ("image", "audio", "video")) == ("image", "audio", "video")
    assert supported_kinds("anthropic", ("image", "audio", "video")) == ("image",)
    assert supported_kinds("openai", ("image", "audio", "video")) == ("image", "audio")


def test_build_openai_audio_content():
    o = build_openai_content("grade", [{"kind": "audio", "mime": "audio/wav", "data": "QUJD"}])
    assert o[1]["type"] == "input_audio" and o[1]["input_audio"]["format"] == "wav"


def test_build_gemini_video_part():
    g = build_gemini_parts("p", [{"kind": "video", "mime": "video/mp4", "url": "gs://b/clip.mp4"}])
    assert g[1]["file_data"]["mime_type"] == "video/mp4"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
