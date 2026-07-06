"""Judge decision logic with stubbed verdicts (no API)."""

import os
import sys
import types

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from shared.judge import resolve_price, _cache_key, LLMJudge, _Cache  # noqa: E402
from shared import reward  # noqa: E402


def _v(deal, price):
    return types.SimpleNamespace(deal_reached=deal, agreed_price=price, reasoning="")


LISTING = 100.0


def test_deal_with_in_band_price_used_directly():
    assert resolve_price(_v(True, 88.0), LISTING) == 88.0


def test_no_deal_is_none():
    assert resolve_price(_v(False, None), LISTING) is None
    # even if the model hallucinated a number on a no-deal, no deal => None
    assert resolve_price(_v(False, 90.0), LISTING) is None


def test_none_verdict_is_none():
    assert resolve_price(None, LISTING) is None


def test_confirmed_deal_with_unusable_price_is_unscorable_never_guessed():
    """A confirmed deal with a null or out-of-band price is JUDGE_FAILED, never a guessed number."""
    assert resolve_price(_v(True, None), LISTING) is reward.JUDGE_FAILED
    assert resolve_price(_v(True, 3.0), LISTING) is reward.JUDGE_FAILED        # below 0.1x sanity band
    assert resolve_price(_v(True, 100000.0), LISTING) is reward.JUDGE_FAILED   # absurdly high
    assert resolve_price(_v(True, 30.0), LISTING) == 30.0    # 0.1x-0.5x is in band, -1 floor stays live


def test_cache_key_stable_and_sensitive():
    turns = [("buyer", "How about $70?"), ("seller", "I'll take 85."), ("buyer", "Okay.")]
    k = _cache_key("gemini-3.1-flash-lite", LISTING, turns)
    assert k == _cache_key("gemini-3.1-flash-lite", LISTING, list(turns))   # replayable
    assert k != _cache_key("gpt-5.4-nano", LISTING, turns)                  # model in key
    assert k != _cache_key("gemini-3.1-flash-lite", 200.0, turns)           # listing in key
    assert k != _cache_key("gemini-3.1-flash-lite", LISTING, turns[:2])     # transcript in key


# --- primary -> backup -> drop escalation ---

class _StubJudge(LLMJudge):
    """LLMJudge with scripted verdicts in place of the genai client."""
    def __init__(self, scripted):
        self.model, self.backup_model = "primary", "backup"
        self.cache = _Cache(None)
        self.api_failures = self.backup_calls = 0
        self._scripted, self.models_called = list(scripted), []

    def _verdict_from(self, transcript, model):
        self.models_called.append(model)
        return self._scripted.pop(0)


_SCEN = {"listing": 100.0, "buyer_target": 80.0, "reserve": 50.0}
_JTR = [("buyer", "80?"), ("seller", "I'll do 90."), ("buyer", "ok deal")]


def _verdict(price=90.0, close=1, deal=True):
    return types.SimpleNamespace(deal_reached=deal, agreed_price=price, close_turn=close, reasoning="r")


def test_judge_primary_success_skips_backup_and_caches():
    j = _StubJudge([_verdict()])
    agreed, close = j(_JTR, _SCEN)
    assert agreed == 90.0 and close == 1
    assert j.models_called == ["primary"] and j.backup_calls == 0
    assert j.cache.get(_cache_key("primary", 100.0, _JTR)) is not None   # replayable


def test_judge_escalates_to_backup_on_primary_null():
    j = _StubJudge([None, _verdict(price=90.0)])          # primary null-parse -> backup verdict
    agreed, _ = j(_JTR, _SCEN)
    assert agreed == 90.0
    assert j.models_called == ["primary", "backup"] and j.backup_calls == 1
    assert j.api_failures == 0


def test_judge_escalates_to_backup_on_unusable_primary_price():
    j = _StubJudge([_verdict(price=None), _verdict(price=85.0, close=1)])
    agreed, close = j(_JTR, _SCEN)
    assert agreed == 85.0 and close == 1
    assert j.models_called == ["primary", "backup"] and j.backup_calls == 1
    assert j.api_failures == 0


def test_judge_drops_when_both_verdicts_unusable_and_does_not_cache():
    j = _StubJudge([_verdict(price=None), _verdict(price=100000.0)])   # both confirmed-deal, no usable price
    agreed, close = j(_JTR, _SCEN)
    assert agreed is reward.JUDGE_FAILED and close is None   # caller drops the sample
    assert j.api_failures == 1 and j.backup_calls == 1
    assert j.cache.get(_cache_key("primary", 100.0, _JTR)) is None   # NOT cached -> retried later


def test_judge_both_fail_returns_sentinel_and_does_not_cache():
    j = _StubJudge([None, None])                          # primary AND backup unavailable
    agreed, close = j(_JTR, _SCEN)
    assert agreed is reward.JUDGE_FAILED and close is None
    assert j.api_failures == 1 and j.backup_calls == 1
    assert j.cache.get(_cache_key("primary", 100.0, _JTR)) is None


def test_backup_no_deal_verdict_is_trusted():
    """A backup no-deal is a usable verdict (None price), not a drop."""
    j = _StubJudge([None, _verdict(deal=False, price=None, close=None)])
    agreed, close = j(_JTR, _SCEN)
    assert agreed is None and close is None
    assert j.api_failures == 0


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} judge tests")


if __name__ == "__main__":
    _run_all()
