"""Price reward and a deterministic agreed-price extractor."""

import re

# Sentinel: judge produced no usable verdict. Not a no-deal — the caller must drop the
# sample, never score it 0.
JUDGE_FAILED = object()

PRICE_BAND = (0.1, 1.5)          # agreed price must fall in this band * listing to count
_ACCEPT_RE = re.compile(
    r"\b(deal|sold|i'?ll take it|i will take it|i'?ll take|take it|works for me|"
    r"sounds good|that works|you'?ve got a deal|you got a deal|it'?s a deal|"
    r"i accept|agreed|let'?s do it|come (?:get|pick)|see you)\b",
    re.I,
)
# $1,200 | 1200 | 1200.50 | 13k | 13 grand
_NUM_RE = re.compile(r"\$?\s?(\d{1,3}(?:,\d{3})+|\d+(?:\.\d{1,2})?)\s?(k|grand)?\b", re.I)


def price_reward(agreed_price, reserve, ceiling):
    """Scale-free price reward in [-1, 1]. None (no deal) -> 0; below reserve -> -1."""
    if agreed_price is None:
        return 0.0
    if agreed_price < reserve:
        return -1.0
    return min(1.0, (agreed_price - reserve) / (ceiling - reserve))


def _prices_in(text, listing):
    """In-band numeric prices in a string, in order. Handles '13k' -> 13000."""
    out = []
    lo, hi = PRICE_BAND[0] * listing, PRICE_BAND[1] * listing
    for m in _NUM_RE.finditer(text):
        val = float(m.group(1).replace(",", ""))
        if m.group(2):                       # 'k' / 'grand'
            val *= 1000.0
        if lo <= val <= hi:
            out.append(val)
    return out


def extract_agreed_price(turns, listing, buyer_target=None):
    """Agreed price from a finished transcript, or None; buyer_target if the close restates no number."""
    if not turns:
        return None

    texts = [t for _, t in turns]
    tail = " ".join(texts[-2:])              # acceptance usually lands in the last turn or two
    if not _ACCEPT_RE.search(tail):
        return None

    for text in reversed(texts):
        prices = _prices_in(text, listing)
        if prices:
            return prices[-1]

    return float(buyer_target) if buyer_target is not None else None


def deterministic_judge(turns, scenario):
    """Extractor as a judge; used only when cfg.judge_model is empty."""
    return extract_agreed_price(turns, scenario["listing"], scenario.get("buyer_target"))
