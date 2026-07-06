"""Opponent buyer: Gemini for training, OpenAI for grading, behind one reply() interface."""

import os
import time

from . import render

_DOTENV_LOADED = False


def _load_env():
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    try:
        from dotenv import load_dotenv, find_dotenv
        load_dotenv(find_dotenv(usecwd=True))
        root_env = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
        if os.path.exists(root_env):
            load_dotenv(root_env)
    except Exception:
        pass
    _DOTENV_LOADED = True


def _truncate(text, max_chars):
    text = (text or "").strip()
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0]
    return text


class Buyer:
    """Builds the no-persona prompt, retries, truncates. Subclasses implement _call."""

    def __init__(self, model, max_chars=500, temperature=1.0, retries=3):
        _load_env()
        self.model = model
        self.max_chars = max_chars
        self.temperature = temperature
        self.retries = retries
        self.failures = 0

    def reply(self, turns, scenario, seed=None):
        """Buyer's next message, or None after `retries` failed attempts (caller drops the sample)."""
        system = render.buyer_prompt(
            scenario["listing"], scenario["title"], scenario["description"], scenario["buyer_target"])
        user = render.render_transcript(turns)
        for attempt in range(self.retries):
            try:
                out = self._call(system, user, seed)
                out = _truncate(out, self.max_chars)
                if out:
                    return out
            except Exception as e:  # noqa: BLE001
                if attempt == self.retries - 1:
                    print(f"  WARN buyer call failed ({self.model}): {e}", flush=True)
                time.sleep(0.5 * (attempt + 1))
        self.failures += 1
        return None

    def _call(self, system, user, seed):
        raise NotImplementedError


class GeminiBuyer(Buyer):
    def __init__(self, model="gemini-2.5-flash", **kw):
        super().__init__(model, **kw)
        from google import genai
        from google.genai import types
        # explicit timeout (ms): the SDK default is no timeout, so a stalled read blocks forever
        self._client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"],
                                    http_options=types.HttpOptions(timeout=120_000))

    def _call(self, system, user, seed):
        from google.genai import types
        cfg = types.GenerateContentConfig(
            system_instruction=system,
            temperature=self.temperature,
            max_output_tokens=256,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            **({"seed": seed} if seed is not None else {}),
        )
        resp = self._client.models.generate_content(model=self.model, contents=user, config=cfg)
        return resp.text


class OpenAIBuyer(Buyer):
    def __init__(self, model="gpt-5.4-nano", temperature=0.0, **kw):
        super().__init__(model, temperature=temperature, **kw)
        from openai import OpenAI
        key = os.environ.get("OPEN_AI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise KeyError("OPEN_AI_API_KEY (or OPENAI_API_KEY) not set for the grade buyer")
        # SDK default timeout is 600s per attempt; 120s keeps a hang within the retry ladder
        self._client = OpenAI(api_key=key, timeout=120.0)

    def _call(self, system, user, seed):
        # temperature not sent: the gpt-5 family rejects non-default values; determinism comes from seed
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            reasoning_effort="none",
            max_completion_tokens=256,
            **({"seed": seed} if seed is not None else {}),
        )
        return resp.choices[0].message.content


def make_buyer(cfg, kind):
    """kind='train' -> Gemini (temp 1.0); kind='grade' -> OpenAI (temp 0.0)."""
    if kind == "train":
        return GeminiBuyer(cfg.train_buyer_model, max_chars=cfg.buyer_max_chars, temperature=1.0)
    if kind == "grade":
        return OpenAIBuyer(cfg.grade_buyer_model, max_chars=cfg.buyer_max_chars, temperature=0.0)
    raise ValueError(f"unknown buyer kind: {kind}")
