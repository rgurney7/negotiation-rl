"""SFT loss masking with a stub tokenizer (no model download)."""

import os
import sys
import types

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from sft.masking import build_labels  # noqa: E402


class StubChatTokenizer:
    """ChatML-ish char-level tokenizer, so prompt is a clean prefix of full."""
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kw):
        # `enable_thinking` etc. accepted and ignored
        s = "".join(f"<|{m['role']}|>{m['content']}<|end|>" for m in messages)
        if add_generation_prompt:
            s += "<|assistant|>"
        return s

    def __call__(self, text, add_special_tokens=False):
        return types.SimpleNamespace(input_ids=[ord(c) for c in text])

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


def test_mask_covers_prompt_only():
    tok = StubChatTokenizer()
    messages = [
        {"role": "system", "content": "SELLER PROMPT"},
        {"role": "user", "content": "Negotiation Transcript:\n[Buyer]: hi\n\n[Your Turn]:"},
        {"role": "assistant", "content": "Sure, I can do $90."},
    ]
    out = build_labels(messages, tok, max_seq_len=10_000)

    trained_ids = [t for t, l in zip(out["input_ids"], out["labels"]) if l != -100]
    masked_ids = [t for t, l in zip(out["input_ids"], out["labels"]) if l == -100]
    trained = tok.decode(trained_ids)
    masked = tok.decode(masked_ids)

    # loss falls on exactly the seller turn + stop token
    assert trained == "Sure, I can do $90.<|end|>"
    assert masked.endswith("<|assistant|>")
    assert "SELLER PROMPT" in masked and "[Buyer]: hi" in masked
    assert "Sure, I can do" not in masked
    assert len(out["input_ids"]) == len(out["labels"]) == len(out["attention_mask"])


def test_prefix_matches_rl_renderer():
    """The masked prefix must decode to exactly shared.model.build_prompt's output."""
    from shared.model import build_prompt
    tok = StubChatTokenizer()
    system, user = "SELLER PROMPT", "Negotiation Transcript:\n[Buyer]: hi\n\n[Your Turn]:"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": "Sure, I can do $90."},
    ]
    rl_prompt = build_prompt(tok, system, user)
    out = build_labels(messages, tok, max_seq_len=10_000)
    masked_ids = [t for t, l in zip(out["input_ids"], out["labels"]) if l == -100]
    assert tok.decode(masked_ids) == rl_prompt


def test_truncation_caps_length():
    tok = StubChatTokenizer()
    messages = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "U"},
        {"role": "assistant", "content": "A" * 100},
    ]
    out = build_labels(messages, tok, max_seq_len=20)
    assert len(out["input_ids"]) == 20
    assert len(out["labels"]) == 20


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} masking tests")


if __name__ == "__main__":
    _run_all()
