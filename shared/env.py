"""Negotiation environment shared by PPO (multi-turn), GRPO (single closing turn), and eval."""

import random

from . import render, reward
from .reward import price_reward


class NegotiationEnv:
    def __init__(self, scenarios, buyer, cfg, single_turn=False, judge=None):
        self.scenarios = scenarios
        self.buyer = buyer
        self.cfg = cfg
        self.single_turn = single_turn
        # judge(turns, scenario) -> price | None, or (price, close_turn) for the LLM judge
        self.judge = judge or reward.deterministic_judge
        self.turns = []
        self.scenario = None
        self.seller_prompt = None
        self.listing_price = None
        self._buyer_seed = None
        self._seed_seller_turns = 0

    def reset(self, seed=None):
        sc = self.scenarios[(seed or 0) % len(self.scenarios)]
        self.scenario = sc
        self.seller_prompt = sc["system"]
        self.listing_price = sc["listing"]
        self.turns = render.parse_seed(sc["seed"])      # recorded buyer opener (+ context for GRPO)
        self.t = 0
        self._buyer_seed = seed
        # seller turns already in the seed, so judge close_turn maps to this episode's step count
        self._seed_seller_turns = sum(1 for r, _ in self.turns if not str(r).lower().startswith("b"))
        return self.obs()

    def get_seller_prompt(self):
        return self.seller_prompt

    def obs(self):
        return render.render_transcript(self.turns)

    def step(self, seller_text):
        # sanitize generated text so an embedded "[Buyer]: ..." can't spoof a real turn
        self.turns.append(("seller", render.sanitize_utterance(seller_text)))
        self.t += 1
        terminal = self.single_turn or (self.t >= self.cfg.num_turns)

        # Buyer replies even on the terminal turn, so the agreed price reflects its final word.
        reply = self.buyer.reply(self.turns, self.scenario, seed=self._buyer_seed)
        if reply is None:
            # buyer unavailable after retries -> unscorable; flag so the caller drops the sample
            return self.obs(), 0.0, True, False, {"buyer_failed": True, "agreed_price": None,
                                                  "listing_price": self.listing_price,
                                                  "scenario": self.scenario["id"]}
        self.turns.append(("buyer", render.sanitize_utterance(reply)))

        if not terminal:
            return self.obs(), 0.0, False, False, {}

        verdict = self.judge(self.turns, self.scenario)
        # LLM judge -> (price, close_turn); deterministic judge -> bare price.
        agreed, close_turn = verdict if isinstance(verdict, tuple) else (verdict, None)
        if agreed is reward.JUDGE_FAILED:
            # no usable verdict -> unscorable, same as a buyer outage; caller drops the sample
            return self.obs(), 0.0, True, False, {"judge_failed": True, "agreed_price": None,
                                                  "listing_price": self.listing_price,
                                                  "scenario": self.scenario["id"]}
        rew = price_reward(agreed, self.scenario["reserve"], self.listing_price)
        # close_step = judge's seller index minus seed seller turns; None -> full horizon
        close_step = None if close_turn is None else max(1, int(close_turn) - self._seed_seller_turns)
        info = {"agreed_price": agreed, "listing_price": self.listing_price,
                "scenario": self.scenario["id"], "reward": rew,
                "close_turn": close_turn, "close_step": close_step}
        # GRPO single step = terminal; PPO horizon cap = truncation. GAE bootstraps from 0 either way.
        return self.obs(), rew, self.single_turn, (not self.single_turn), info


def make_envs(cfg, scenarios, buyer, n, single_turn=False, judge=None):
    """n independent envs over a scenario pool shuffled deterministically by cfg.seed."""
    pool = list(scenarios)
    random.Random(cfg.seed).shuffle(pool)
    return [NegotiationEnv(pool, buyer, cfg, single_turn=single_turn, judge=judge) for _ in range(n)]
