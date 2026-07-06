"""Deterministic helpers shared by enrich_llm.py and enrich.py."""
import hashlib
import re

import requests
import urllib3

# SSL shim: the upstream loader reaches a host with an expired cert.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
_ORIG_REQUEST = requests.Session.request
requests.Session.request = lambda self, *a, **k: _ORIG_REQUEST(self, *a, **{**k, "verify": False})

from datasets import load_dataset  # noqa: E402

# ----------------------------------------------------------------------------- config
HF_REPO = "stanfordnlp/craigslist_bargains"
HF_REVISION = "cfb6992c5ca9bad209323ed8e42e0cfc7e4178cf"  # pinned for reproducibility

SENT = -1.0                       # dialogue_acts price sentinel = "no price"
ACTION_INTENTS = {"offer", "accept", "reject", "quit"}  # AMT UI button-clicks (always empty)

# golden gate thresholds
RATIO_BAND = (0.5, 1.5)           # agreed/listing; floor = reward reserve (0.5*listing -> -1)
PRICE_BAND = (0.1, 1.5)           # a "priced move" must fall in this band * listing
MIN_CONTENTFUL_TURNS = 4
MIN_SELLER_PRICED = 2
MIN_BUYER_PRICED = 1
REWARD_FLOOR_FRAC = 0.5           # reserve = 0.5 * listing (matches shared/reward.py)

# shared prompt format, identical across methods
TRANSCRIPT_HEADER = "Negotiation Transcript:"
TURN_MARKER = "[Your Turn]:"
MAX_DESC_CHARS = 400


def load_hf():
    return load_dataset(HF_REPO, revision=HF_REVISION, trust_remote_code=True)


# ----------------------------------------------------------------------------- ids
def _norm(s):
    return re.sub(r"\s+", " ", (s or "").strip()).lower()


def dialogue_id(role_text_pairs):
    """Stable id from the normalized transcript."""
    h = hashlib.sha1()
    for role, text in role_text_pairs:
        h.update(f"{role[0]}:{_norm(text)}\n".encode())
    return h.hexdigest()[:16]


def scenario_id(listing, buyer_target, title, description):
    """Groups dialogues that share a scenario."""
    key = f"{listing:.2f}|{buyer_target:.2f}|{_norm(title)}|{_norm(description)}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]


# ----------------------------------------------------------------------------- model
class Turn:
    __slots__ = ("idx", "role", "intent", "price", "text", "is_empty", "is_action_marker")

    def __init__(self, idx, role, intent, price, text, is_empty, is_action_marker):
        self.idx, self.role, self.intent, self.price = idx, role, intent, price
        self.text, self.is_empty, self.is_action_marker = text, is_empty, is_action_marker

    def as_dict(self):
        return {"idx": self.idx, "role": self.role, "intent": self.intent,
                "price": self.price, "text": self.text, "is_empty": self.is_empty,
                "is_action_marker": self.is_action_marker}


class Dialogue:
    __slots__ = ("did", "sid", "split", "listing", "buyer_target", "seller_target",
                 "title", "description", "category", "turns", "bi", "si")

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

    @property
    def reserve(self):
        return REWARD_FLOOR_FRAC * self.listing


def parse_dialogue(e, split):
    """Raw HF example -> Dialogue, or None if malformed."""
    roles = list(e["agent_info"]["Role"])
    try:
        bi, si = roles.index("buyer"), roles.index("seller")
    except ValueError:
        return None
    targets = list(e["agent_info"]["Target"])
    it = e["items"]
    listing = float(it["Price"][0])
    if listing <= 0:
        return None
    ints, prc = e["dialogue_acts"]["intent"], e["dialogue_acts"]["price"]
    utts, aturn = e["utterance"], e["agent_turn"]

    turns, role_text = [], []
    for k in range(len(utts)):
        role = roles[aturn[k]]
        text = (utts[k] or "").strip()
        intent = ints[k] or ""
        price = None if prc[k] == SENT else float(prc[k])
        turns.append(Turn(k, role, intent, price, text,
                          is_empty=(text == ""), is_action_marker=(intent in ACTION_INTENTS)))
        role_text.append((role, text))

    return Dialogue(
        did=dialogue_id(role_text),
        sid=scenario_id(listing, float(targets[bi]), it["Title"][0], it["Description"][0]),
        split=split, listing=listing, buyer_target=float(targets[bi]),
        seller_target=float(targets[si]), title=it["Title"][0],
        description=it["Description"][0], category=it["Category"][0],
        turns=turns, bi=bi, si=si,
    )


def iter_dialogues(ds, split):
    seen = set()
    for e in ds[split]:
        d = parse_dialogue(e, split)
        if d is None or d.did in seen:   # skip malformed + exact-duplicate transcripts
            continue
        seen.add(d.did)
        yield d


# ----------------------------------------------------------------------------- gate
def acts_outcome(d):
    """Deal + agreed price derived from the dialogue acts."""
    ints = [t.intent for t in d.turns]
    if "accept" not in ints:
        return {"deal": False, "agreed": None, "accept_idx": None}
    ai = ints.index("accept")
    offers = [d.turns[j].price for j in range(ai + 1)
              if d.turns[j].intent == "offer" and d.turns[j].price is not None]
    anyp = [d.turns[j].price for j in range(ai + 1) if d.turns[j].price is not None]
    agreed = offers[-1] if offers else (anyp[-1] if anyp else None)
    return {"deal": True, "agreed": agreed, "accept_idx": ai}


def priced_moves(d, role):
    """Contentful (non-marker) turns by `role` carrying an in-band price."""
    lo, hi = PRICE_BAND[0] * d.listing, PRICE_BAND[1] * d.listing
    return sum(1 for t in d.turns
               if t.role == role and not t.is_empty and t.price is not None and lo <= t.price <= hi)


def golden_gate(d):
    """Deterministic structural + outcome gate. Returns (is_golden, reason, outcome)."""
    o = acts_outcome(d)
    if not o["deal"] or o["agreed"] is None:
        return False, "no_deal", o
    if not (d.listing > d.buyer_target):
        return False, "listing_le_buyer_target", o
    ratio = o["agreed"] / d.listing
    if not (RATIO_BAND[0] <= ratio <= RATIO_BAND[1]):
        return False, "ratio_out_of_band", o
    if sum(1 for t in d.turns if not t.is_empty) < MIN_CONTENTFUL_TURNS:
        return False, "too_short", o
    if priced_moves(d, "seller") < MIN_SELLER_PRICED:
        return False, "seller_underpriced", o
    if priced_moves(d, "buyer") < MIN_BUYER_PRICED:
        return False, "buyer_underpriced", o
    return True, "golden", o


# ----------------------------------------------------------------------------- prompt + render
def item_description(d):
    """Cleaned, length-capped item description (shown to both sides)."""
    desc = re.sub(r"\s+", " ", (d.description or "").strip())
    if len(desc) > MAX_DESC_CHARS:
        desc = desc[:MAX_DESC_CHARS].rsplit(" ", 1)[0] + "..."
    return desc


def seller_prompt(d):
    desc = item_description(d)
    return (
        "You are a seller on Craigslist. Your goal is to maximize the sale price while still "
        "closing the deal.\n"
        f"You listed this item at ${d.listing:.0f}.\n\n"
        f"Item: {d.title.strip()}\n"
        f"Description: {desc}\n\n"
        "Negotiate on price only. Do not offer extras, add-ons, free items, delivery, warranties, "
        "or anything beyond the item itself.\n\n"
        "Write your next message only. One to three sentences of natural dialogue. Do not start "
        "your message with any label or prefix. Do not write the buyer's response."
    )


def render_context(d, upto_idx, from_idx=0):
    """User message: contentful turns with from_idx <= idx < upto_idx, then [Your Turn]:."""
    lines = []
    for t in d.turns:
        if t.is_empty or t.idx >= upto_idx or t.idx < from_idx:
            continue
        spk = "Buyer" if t.role == "buyer" else "Seller"
        lines.append(f"[{spk}]: {t.text}")
    body = "\n".join(lines)
    return f"{TRANSCRIPT_HEADER}\n{body}\n\n{TURN_MARKER}"


def render_for_llm(d):
    """Full dialogue with original turn indices; input for Pass A/B."""
    out = []
    for t in d.turns:
        txt = t.text if not t.is_empty else "(empty - formal marker)"
        out.append(f"[{t.idx}] {t.role.upper()}: {txt}")
    return "\n".join(out)


def first_buyer_msg_index(d):
    for t in d.turns:
        if t.role == "buyer" and not t.is_empty:
            return t.idx
    return None


def last_contentful_seller_index(d):
    idxs = [t.idx for t in d.turns if t.role == "seller" and not t.is_empty]
    return idxs[-1] if idxs else None


def price_in_context(d, upto_idx, price, tol=0.51):
    """Is `price` present (as an act-price or a number in text) in any contentful turn before upto_idx?"""
    if price is None:
        return True
    for t in d.turns:
        if t.is_empty or t.idx >= upto_idx:
            continue
        if t.price is not None and abs(t.price - price) <= tol:
            return True
        for m in re.finditer(r"\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?", t.text):
            try:
                if abs(float(m.group(0).replace(",", "")) - price) <= tol:
                    return True
            except ValueError:
                pass
    return False
