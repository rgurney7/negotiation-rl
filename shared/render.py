"""Transcript and prompt formatting."""

import re

# Must equal data/enrich_common.py.
TRANSCRIPT_HEADER = "Negotiation Transcript:"
TURN_MARKER = "[Your Turn]:"
MAX_DESC_CHARS = 400

# Markers embedded in generated text would read as real turns to the buyer and the judge, so
# strip both render formats from generated utterances. Frozen seeds are not sanitized: their
# markers are the real structure parse_seed consumes.
_MARKER_RE = re.compile(r"\[\s*(?:Buyer|Seller|Your\s+Turn)\s*\]\s*:?\s*", flags=re.IGNORECASE)
_JUDGE_MARKER_RE = re.compile(r"\b(?:BUYER|SELLER)\s*(?:\[\s*\d+\s*\])?\s*:\s*", flags=re.IGNORECASE)
_WS_RUN_RE = re.compile(r"\s{2,}")


def sanitize_utterance(text):
    """Strip role/turn markers (both render formats) from generated text and flatten to one line."""
    if not text:
        return text
    text = " ".join(text.splitlines())          # every line-break class -> a single space
    while True:                                  # markers of both formats, to a fixpoint
        stripped = _JUDGE_MARKER_RE.sub("", _MARKER_RE.sub("", text))
        if stripped == text:
            break
        text = stripped
    return _WS_RUN_RE.sub(" ", text).strip()


# The SFT policy fails to stop at its turn boundary and streams a fake continuation of the
# dialogue ("user ... assistant <think> ...") that the buyer then reads and sometimes prices
# from (see results/eval_s*/sft_eval.json — every SFT episode carries it). These are the
# markers that continuation starts with; nothing before the first one is template text.
_LEAK_PATTERNS = [
    re.compile(r"<\|im_(?:start|end)\|>"),
    re.compile(r"</?think>", flags=re.IGNORECASE),
    re.compile(r"(?<![A-Za-z])(?:user|assistant)(?![A-Za-z])"),
    re.compile(re.escape(TRANSCRIPT_HEADER)),
]


def truncate_template_leak(text):
    """Cut generated text at the first chat-template continuation marker, keeping only the
    genuine reply. Used by the sanitized eval variant (eval_methods --sanitize-leak); the
    unsanitized path is what the published eval_s* results ran."""
    if not text:
        return text
    cut = len(text)
    for pat in _LEAK_PATTERNS:
        m = pat.search(text)
        if m:
            cut = min(cut, m.start())
    return text[:cut].strip()


def _spk(role):
    return "Buyer" if str(role).lower().startswith("b") else "Seller"


def render_transcript(turns):
    """[(role, text), ...] -> observation string ending with the turn marker."""
    body = "\n".join(f"[{_spk(r)}]: {t}" for r, t in turns)
    return f"{TRANSCRIPT_HEADER}\n{body}\n\n{TURN_MARKER}"


def parse_seed(seed_text):
    """Inverse of render_transcript -> [(role, text), ...]."""
    body = seed_text
    if body.startswith(TRANSCRIPT_HEADER):
        body = body[len(TRANSCRIPT_HEADER):]
    idx = body.rfind(TURN_MARKER)
    if idx != -1:
        body = body[:idx]
    body = body.strip("\n")

    turns, role, lines = [], None, []
    for line in body.split("\n"):
        if line.startswith("[Buyer]: "):
            if role is not None:
                turns.append((role, "\n".join(lines)))
            role, lines = "buyer", [line[len("[Buyer]: "):]]
        elif line.startswith("[Seller]: "):
            if role is not None:
                turns.append((role, "\n".join(lines)))
            role, lines = "seller", [line[len("[Seller]: "):]]
        elif role is not None:
            lines.append(line)
    if role is not None:
        turns.append((role, "\n".join(lines)))
    return turns


def seller_prompt(listing, title, description):
    """Seller system prompt. `description` must be the already-cleaned stored field; don't re-clean."""
    desc = description
    return (
        "You are a seller on Craigslist. Your goal is to maximize the sale price while still "
        "closing the deal.\n"
        f"You listed this item at ${listing:.0f}.\n\n"
        f"Item: {title.strip()}\n"
        f"Description: {desc}\n\n"
        "Negotiate on price only. Do not offer extras, add-ons, free items, delivery, warranties, "
        "or anything beyond the item itself.\n\n"
        "Write your next message only. One to three sentences of natural dialogue. Do not start "
        "your message with any label or prefix. Do not write the buyer's response."
    )


def buyer_prompt(listing, title, description, buyer_target):
    """Buyer system prompt for the opponent, anchored to buyer_target."""
    desc = description
    return (
        "You are a buyer on Craigslist. You are interested in this item and are negotiating to "
        "buy it for as low a price as you reasonably can.\n"
        f"The seller listed it at ${listing:.0f}. You are hoping to pay about ${buyer_target:.0f}, "
        "and you should not agree to pay much more than that.\n\n"
        f"Item: {title.strip()}\n"
        f"Description: {desc}\n\n"
        "Negotiate on price only. When you reach a price you are happy with, accept the deal "
        "clearly.\n\n"
        "Write your next message only. One to three sentences of natural dialogue. Do not start "
        "your message with any label or prefix. Do not write the seller's response."
    )
