"""truncate_template_leak: cut generated text at the first chat-template continuation marker."""

import os
import sys

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from shared.render import truncate_template_leak  # noqa: E402


def test_cuts_at_role_word():
    raw = "I can do $175 if you pick it up today. user I can do $150. assistant Okay, deal."
    assert truncate_template_leak(raw) == "I can do $175 if you pick it up today."


def test_cuts_at_think_block():
    raw = "I'm asking $40 for it.\nuser\nWould you take $30?\nassistant\n<think>\n\n</think>\n$35"
    assert truncate_template_leak(raw) == "I'm asking $40 for it."
    assert truncate_template_leak("<think>\n\n</think>Sure, $35 works.") == ""


def test_cuts_at_transcript_header_echo():
    raw = "Sounds good. Negotiation Transcript:\n[Buyer]: hi there"
    assert truncate_template_leak(raw) == "Sounds good."


def test_cuts_at_special_tokens():
    raw = "That works for me.<|im_end|>\n<|im_start|>user\nGreat!"
    assert truncate_template_leak(raw) == "That works for me."


def test_earliest_marker_wins():
    raw = "Yes. user offers assistant replies <think> more"
    assert truncate_template_leak(raw) == "Yes."


def test_clean_text_untouched():
    for clean in (
        "I can meet you at $420, and that's as low as I can go.",
        "It's in great condition — the previous owner barely used it.",   # 'used' != 'user'
        "",
    ):
        assert truncate_template_leak(clean) == clean


if __name__ == "__main__":
    test_cuts_at_role_word()
    test_cuts_at_think_block()
    test_cuts_at_transcript_header_echo()
    test_cuts_at_special_tokens()
    test_earliest_marker_wins()
    test_clean_text_untouched()
    print("OK")
