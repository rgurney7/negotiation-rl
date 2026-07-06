"""LLM enrichment: Pass A (concession detection) + Pass B (closing-turn truncation) over golden candidates."""
import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

import enrich_common as ec

load_dotenv(os.path.join(os.path.dirname(__file__), os.pardir, ".env"))
from google import genai            # noqa: E402
from google.genai import types      # noqa: E402

MODEL = "gemini-3.1-flash-lite-preview"
CACHE_PATH = os.path.join(os.path.dirname(__file__), "cache", "llm_enrichment.jsonl")
_client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"],
                       http_options=types.HttpOptions(timeout=60_000))


class RateLimiter:
    """Evenly spaces request starts to stay under `rpm`."""
    def __init__(self, rpm):
        self.interval = 60.0 / rpm
        self.lock = threading.Lock()
        self.next_at = 0.0

    def acquire(self):
        with self.lock:
            now = time.monotonic()
            t = max(now, self.next_at)
            self.next_at = t + self.interval
            wait = t - now
        if wait > 0:
            time.sleep(wait)


_limiter = RateLimiter(14)  # overridden by --rpm in main()


class Concession(BaseModel):
    turn_index: int = Field(description="index of the SELLER turn that makes the concession")
    concession_type: Literal["delivery", "bundle_extra", "free_item", "warranty", "other"]
    quote: str = Field(description="exact phrase from that turn")
    reasoning: str = Field(description="one short sentence")


class ConcessionResult(BaseModel):
    concessions: list[Concession]


class TruncationResult(BaseModel):
    close_turn_index: int = Field(description="SELLER turn that locks the final price; -1 if no deal")
    agreed_price: Optional[float]
    close_type: Literal["seller_proposes", "seller_accepts", "no_deal", "other"]
    renegotiated_after_first_agreement: bool
    reasoning: str


PASS_A_SYS = """You audit a Craigslist price-negotiation transcript. The SELLER was instructed to negotiate on PRICE ONLY and never offer anything beyond the item itself. Find every SELLER turn where the seller offers, adds, throws in, or agrees to give the buyer something of value BEYOND the listed item at the agreed price (a "concession").

FLAG (concession):
- Free or discounted delivery / shipping / dropping the item at the buyer's location
- Throwing in / including / adding an extra physical item (accessory, second product, supplies)
- A free gift, bonus, freebie, "on the house" item
- A warranty, guarantee, or return policy offered as a sweetener
- Any "I'll also..." handing over value other than a lower price

DO NOT FLAG (fine):
- Adjusting the PRICE itself (lowering, "meet in the middle", a discount as a number)
- Meeting at a neutral/public spot or "meet halfway" for handoff (standard logistics, no cost borne)
- Describing what the item already includes ("comes with the original charger it shipped with")
- The BUYER requesting an extra unless the SELLER agrees to / offers it
Only flag SELLER turns, using the exact indices shown. If none, return an empty list.

EXAMPLES:
[0] BUYER: Interested in the desk. Would you take $80?
[1] SELLER: I can do $90, and I'll deliver it to your place for free.
[2] BUYER: Deal at $90.
-> {"concessions":[{"turn_index":1,"concession_type":"delivery","quote":"I'll deliver it to your place for free","reasoning":"free delivery is value beyond the item"}]}

[0] BUYER: $200 for the bike?
[1] SELLER: Barely used. I can come down to $230 and we can meet at the coffee shop on Main.
[2] BUYER: $230 works.
-> {"concessions":[]}

[0] BUYER: $300 is my max for the camera.
[1] SELLER: For $340 I'll throw in the spare battery and the carry case.
-> {"concessions":[{"turn_index":1,"concession_type":"bundle_extra","quote":"I'll throw in the spare battery and the carry case","reasoning":"extra items bundled beyond the listed camera"}]}"""

PASS_B_SYS = """You prepare a Craigslist negotiation transcript for reinforcement learning. Return the single SELLER turn that CLOSES the deal: the turn where the seller commits to the price that becomes the FINAL agreed price, either by (a) PROPOSING that price (buyer then accepts) or (b) ACCEPTING the buyer's offer of that price in their own words.

RULES:
- close_turn_index MUST be a SELLER turn containing real text. NEVER pick an empty turn; the dataset ends with empty formal "offer"/"accept" markers - ignore those.
- The closing turn must lock the FINAL price. If the parties verbally agree, then keep haggling and settle on a DIFFERENT final price, pick the LATER seller turn that commits to the actual final price, and set renegotiated_after_first_agreement=true.
- agreed_price = the final sale price. If no agreement, close_turn_index=-1, close_type="no_deal".

EXAMPLES:
[0] BUYER: I'm willing to pay $114
[1] SELLER: 114 is too low, how about 130?
[2] BUYER: I can do 120
[3] SELLER: ok, 125 and we have a deal
[4] BUYER: sounds good, 125
[5] BUYER: (empty - formal marker)
[6] SELLER: (empty - formal marker)
-> {"close_turn_index":3,"agreed_price":125,"close_type":"seller_proposes","renegotiated_after_first_agreement":false,"reasoning":"seller proposes 125, buyer accepts"}

[0] BUYER: would you take 85?
[1] SELLER: It's unlocked and like new. 85 works, it's a deal.
[2] BUYER: (empty - formal marker)
[3] SELLER: (empty - formal marker)
-> {"close_turn_index":1,"agreed_price":85,"close_type":"seller_accepts","renegotiated_after_first_agreement":false,"reasoning":"seller accepts the buyer's 85 in words"}

[0] BUYER: 100?
[1] SELLER: I could do 120.
[2] BUYER: ok 120 sounds fine.
[3] BUYER: actually, can you meet me at 110?
[4] SELLER: alright, 110 it is.
[5] BUYER: (empty - formal marker)
[6] SELLER: (empty - formal marker)
-> {"close_turn_index":4,"agreed_price":110,"close_type":"seller_accepts","renegotiated_after_first_agreement":true,"reasoning":"first agreed 120 then reopened and settled at 110"}"""


def _retry(fn, tries=8):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            if i == tries - 1:
                raise
            msg = str(e)
            delay = min(2 ** i, 30)
            m = re.search(r"retryDelay['\"]?:?\s*['\"]?(\d+)", msg)  # honor the server's 429 hint
            if m:
                delay = int(m.group(1)) + 1
            time.sleep(delay)


def _call(system, user, schema):
    def once():
        _limiter.acquire()
        return _client.models.generate_content(
            model=MODEL, contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system, temperature=0,
                response_mime_type="application/json", response_schema=schema))
    return _retry(once).parsed


def enrich_one(d):
    """Run both passes on one Dialogue; returns the cache record, or None on failure."""
    text = ec.render_for_llm(d)
    a = _call(PASS_A_SYS, text, ConcessionResult)
    b = _call(PASS_B_SYS, text, TruncationResult)
    if a is None or b is None:
        return None
    acts_agreed = ec.acts_outcome(d)["agreed"]
    return {
        "did": d.did,
        "concessions": [c.model_dump() for c in a.concessions],
        "close_turn_index": b.close_turn_index,
        "close_type": b.close_type,
        "llm_agreed_price": b.agreed_price,
        "renegotiated": b.renegotiated_after_first_agreement,
        "pass_b_reasoning": b.reasoning,
        "acts_agreed_price": acts_agreed,
        "agreed_mismatch": (b.agreed_price is None or acts_agreed is None
                            or abs(b.agreed_price - acts_agreed) > 0.51),
    }


def load_cache():
    done = {}
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)           # tolerate a partial trailing line
                except json.JSONDecodeError:
                    continue
                done[rec["did"]] = rec
    return done


def main():
    global _limiter
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--rpm", type=int, default=14, help="requests/min cap (free tier=15)")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    _limiter = RateLimiter(args.rpm)

    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    done = load_cache()
    ds = ec.load_hf()
    todo = []
    for d in ec.iter_dialogues(ds, "train"):
        ok, _, _ = ec.golden_gate(d)
        if ok and d.did not in done:
            todo.append(d)
    if args.limit:
        todo = todo[:args.limit]
    print(f"cached: {len(done)} | to enrich: {len(todo)}", flush=True)

    lock = threading.Lock()
    n_ok = n_fail = 0
    with open(CACHE_PATH, "a") as cache_f, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(enrich_one, d): d for d in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                rec = fut.result()
            except Exception as e:  # noqa: BLE001
                rec = None
                print(f"  ERROR {futs[fut].did}: {str(e)[:90]}", flush=True)
            if rec is None:
                n_fail += 1
                continue
            with lock:
                cache_f.write(json.dumps(rec) + "\n")
                cache_f.flush()
            n_ok += 1
            if i % 100 == 0:
                print(f"  {i}/{len(todo)} done ({n_ok} ok, {n_fail} fail)", flush=True)
    print(f"DONE: {n_ok} enriched, {n_fail} failed. cache -> {CACHE_PATH}", flush=True)


if __name__ == "__main__":
    main()
