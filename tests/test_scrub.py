"""PII/secret scrubbing (regex baseline) — offline, no deps.

Run:  ../.workers-venv/Scripts/python.exe tests/test_scrub.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jobs.calibration.scrub import scrub_text, scrub_value


# use_presidio=False for exact assertions — the regex baseline is deterministic
# regardless of whether Presidio happens to be installed.

def test_scrub_common_pii():
    text = "Email me at jane.doe@example.com or call (415) 555-1234."
    scrubbed, ents = scrub_text(text, use_presidio=False)
    assert "jane.doe@example.com" not in scrubbed
    assert "555-1234" not in scrubbed
    assert "<EMAIL_ADDRESS>" in scrubbed and "<PHONE_NUMBER>" in scrubbed
    assert set(ents) == {"EMAIL_ADDRESS", "PHONE_NUMBER"}


def test_scrub_secrets_and_ssn_cc():
    text = ("key sk-ant-abcdefghijklmnopqrstuvwxyz0123456789 ssn 123-45-6789 "
            "card 4111111111111111 ip 10.0.0.5")
    scrubbed, ents = scrub_text(text, use_presidio=False)
    assert "sk-ant-abcdefghij" not in scrubbed
    assert "123-45-6789" not in scrubbed
    assert "4111111111111111" not in scrubbed
    assert {"ANTHROPIC_API_KEY", "US_SSN", "CREDIT_CARD", "IP_ADDRESS"} <= set(ents)


def test_scrub_preserves_semantic_content():
    # Locations / names must survive — they are the content of a golden case.
    text = "The capital of France is Paris, home of the Eiffel Tower."
    scrubbed, ents = scrub_text(text, use_presidio=False)
    assert scrubbed == text and ents == []


def test_scrub_value_recurses():
    obj = {
        "messages": [{"role": "user", "content": "reach me at bob@corp.io"}],
        "response": "Sure, I'll email bob@corp.io.",
        "count": 3,
    }
    scrubbed, ents = scrub_value(obj, use_presidio=False)
    assert "bob@corp.io" not in str(scrubbed)
    assert scrubbed["count"] == 3
    assert ents == ["EMAIL_ADDRESS"]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
