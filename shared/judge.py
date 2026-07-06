"""LLM reward judge: reads a finished negotiation, decides deal/no-deal and the final price."""

import hashlib
import json
import os
import re
import threading
import time

from . import render, reward

PRICE_BAND = (0.1, 1.5)

JUDGE_SYS = """You read a FINISHED Craigslist price negotiation between a BUYER and a SELLER. The SELLER's turns are numbered SELLER[1], SELLER[2], ... in order. Decide whether they reached a deal by the end of the conversation, the FINAL agreed price, and the SELLER turn at which the deal became final.

RULES:
- deal_reached = true ONLY if one party clearly accepts a price the other put forward — explicitly ("deal", "I'll take it at 90", "90 works") or by clear implicit acceptance (agreeing to meet/pay/pick up at a settled price).
- If they only discussed or proposed prices without ever agreeing, or someone refused or left it open ("I'll think about it"), deal_reached = false, agreed_price = null, close_turn = null.
- agreed_price = the final number both sides settled on. If they "meet in the middle" or split the difference, compute that number even if it is not written out. Use only prices grounded in the transcript.
- If a price was agreed, then reopened and re-settled at a different number, use the LATER final number.
- close_turn = the index k of the SELLER[k] turn at which the deal became FINAL: the seller offer the buyer then accepts, or the seller turn that accepts the buyer's offer. If the deal was reopened and re-settled, use the LATER seller turn. Seller turns AFTER close_turn are post-deal (confirming, logistics) and do not change close_turn. null if no deal.

EXAMPLES:
BUYER: Would you take $80 for the desk?
SELLER[1]: I can do 90.
BUYER: Deal, 90 works.
-> {"deal_reached": true, "agreed_price": 90, "close_turn": 1, "reasoning": "buyer accepts SELLER[1]'s 90"}

BUYER: Would you take $80?
SELLER[1]: No, lowest I can go is 95.
BUYER: That's more than I want to spend, I'll think about it.
-> {"deal_reached": false, "agreed_price": null, "close_turn": null, "reasoning": "prices discussed but no acceptance"}

BUYER: 100 for the bike?
SELLER[1]: I was hoping for 140. How about we meet in the middle?
BUYER: Okay, that works, see you Saturday.
-> {"deal_reached": true, "agreed_price": 120, "close_turn": 1, "reasoning": "split 100 and 140; buyer agrees after SELLER[1]"}

BUYER: I'll give you 50.
SELLER[1]: Deal, 50 works.
BUYER: Great, can I pick up tomorrow?
SELLER[2]: Yes, tomorrow afternoon is perfect.
-> {"deal_reached": true, "agreed_price": 50, "close_turn": 1, "reasoning": "deal sealed at SELLER[1]=50; SELLER[2] is post-deal logistics"}

BUYER: I could do 200.
SELLER[1]: Sure, 200 and it's yours.
BUYER: Actually can you do 180?
SELLER[2]: Fine, 180, come grab it.
-> {"deal_reached": true, "agreed_price": 180, "close_turn": 2, "reasoning": "reopened after SELLER[1]=200, re-settled at SELLER[2]=180"}"""

# Folded into the cache key, so editing the prompt invalidates old cached verdicts.
_JUDGE_SYS_VER = hashlib.sha1(JUDGE_SYS.encode()).hexdigest()[:8]
# bump on any change to verdict->price resolution; cached records store resolved prices
_CACHE_VER = "3"


def _verdict_schema():
    from typing import Optional
    from pydantic import BaseModel

    class Verdict(BaseModel):
        deal_reached: bool
        agreed_price: Optional[float]
        close_turn: Optional[int]
        reasoning: str
    return Verdict


def _render_for_judge(turns):
    """SELLER turns numbered SELLER[1], SELLER[2], ...; BUYER turns unnumbered."""
    out, k = [], 0
    for r, t in turns:
        if str(r).lower().startswith("b"):
            out.append(f"BUYER: {t}")
        else:
            k += 1
            out.append(f"SELLER[{k}]: {t}")
    return "\n".join(out)


def _cache_key(model, listing, turns):
    """listing is in the key because the band check resolves the price against it."""
    return hashlib.sha1(
        f"{model}|{_JUDGE_SYS_VER}|{_CACHE_VER}|{listing:.2f}|{_render_for_judge(turns)}"
        .encode()).hexdigest()[:16]


def resolve_price(verdict, listing):
    """Verdict -> price. No deal -> None; confirmed deal with a null/out-of-band price -> JUDGE_FAILED."""
    if verdict is None or not verdict.deal_reached:
        return None
    p = verdict.agreed_price
    lo, hi = PRICE_BAND[0] * listing, PRICE_BAND[1] * listing
    if p is not None and lo <= p <= hi:
        return p
    return reward.JUDGE_FAILED


def _resolve_close_turn(verdict, agreed, turns):
    """Validated 1-based seller close index; None (no deal / bad index) keeps the full horizon."""
    if agreed is None or verdict is None:
        return None
    n_seller = sum(1 for r, _ in turns if not str(r).lower().startswith("b"))
    ct = getattr(verdict, "close_turn", None)
    if isinstance(ct, bool):                       # bool is an int subclass; reject it
        return None
    if isinstance(ct, (int, float)) and 1 <= int(ct) <= n_seller:
        return int(ct)
    return None


class _Cache:
    """Thread-safe append-only JSONL cache."""
    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        self.mem = {}
        if path and os.path.exists(path):
            with open(path) as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        self.mem[rec["k"]] = rec["v"]
                    except Exception:
                        pass

    def get(self, key):
        return self.mem.get(key)

    def put(self, key, value):
        with self.lock:
            if key in self.mem:
                return
            self.mem[key] = value
            if self.path:
                os.makedirs(os.path.dirname(self.path), exist_ok=True)
                with open(self.path, "a") as f:
                    f.write(json.dumps({"k": key, "v": value}) + "\n")


class LLMJudge:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = cfg.judge_model
        self.backup_model = getattr(cfg, "judge_backup_model", "")
        self.cache = _Cache(getattr(cfg, "judge_cache_path", None))
        from google import genai
        from google.genai import types
        self._types = types
        # explicit timeout (ms): the SDK default is no timeout, so a stalled read blocks forever
        self._client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"],
                                    http_options=types.HttpOptions(timeout=120_000))
        self._schema = _verdict_schema()
        self.api_failures = 0      # samples DROPPED because both judges failed
        self.backup_calls = 0      # times the backup judge was tried (primary failed)

    def _verdict_from(self, transcript, model):
        """Parsed Verdict, or None (raised after retries, or the SDK set .parsed=None without raising)."""
        def once():
            # thinking_budget=0: a thinking model burns the 512-token cap on reasoning and returns no JSON
            return self._client.models.generate_content(
                model=model, contents=transcript,
                config=self._types.GenerateContentConfig(
                    system_instruction=JUDGE_SYS, temperature=0, max_output_tokens=512,
                    thinking_config=self._types.ThinkingConfig(thinking_budget=0),
                    response_mime_type="application/json", response_schema=self._schema))
        try:
            resp = _retry(once)
        except Exception as e:                       # API down after retries
            print(f"  WARN judge raised ({model}): {str(e)[:80]}", flush=True)
            return None
        v = getattr(resp, "parsed", None)
        if v is None:
            print(f"  WARN judge returned no parse ({model}) -> escalate", flush=True)
        return v

    def __call__(self, turns, scenario):
        """Returns (agreed_price, close_turn); (reward.JUDGE_FAILED, None) when neither judge was usable."""
        listing = scenario["listing"]
        key = _cache_key(self.model, listing, turns)
        cached = self.cache.get(key)
        if cached is not None:
            return cached["agreed_price"], cached.get("close_turn")

        def usable(v):
            # None = judge unavailable/unparsed; JUDGE_FAILED = confirmed deal, unusable price.
            return v is not None and resolve_price(v, listing) is not reward.JUDGE_FAILED

        transcript = _render_for_judge(turns)
        v, used = self._verdict_from(transcript, self.model), self.model
        if not usable(v) and self.backup_model:      # escalate to the backup
            self.backup_calls += 1
            v, used = self._verdict_from(transcript, self.backup_model), self.backup_model
        if not usable(v):                            # unscorable -> drop (no cache, retried later)
            self.api_failures += 1
            print("  WARN judge unusable (primary+backup) -> DROP sample", flush=True)
            return reward.JUDGE_FAILED, None

        agreed = resolve_price(v, listing)
        close_turn = _resolve_close_turn(v, agreed, turns)
        self.cache.put(key, {"agreed_price": agreed, "close_turn": close_turn,
                             "reasoning": v.reasoning, "model": used})
        return agreed, close_turn


def _retry(fn, tries=6):
    """Exponential backoff, honoring a 429 retryDelay hint."""
    for i in range(tries):
        try:
            return fn()
        except Exception as e:                       # noqa: BLE001
            if i == tries - 1:
                raise
            delay = min(2 ** i, 30)
            m = re.search(r"retryDelay['\"]?:?\s*['\"]?(\d+)", str(e))
            if m:
                # cap the server hint; quota errors suggest up to 3600s
                delay = min(int(m.group(1)) + 1, 60)
            time.sleep(delay)


def make_judge(cfg):
    """Empty cfg.judge_model -> deterministic judge; a set model with a missing key raises."""
    if not getattr(cfg, "judge_model", ""):
        return reward.deterministic_judge
    return LLMJudge(cfg)
