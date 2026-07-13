# negotiation-rl

Most RL environments for LLMs today are built for math and code, where a checker verifies the
answer and the best policy is to make every output as correct as possible. Negotiation is a
different kind of task, and an underexplored one. It is still checkable (the agreed price is a
scalar you can score against the listing), but it is open-ended and dynamic: the model faces an
adversarial counterparty rather than a static problem, there is no single correct move, and the
reward arrives at the end of a conversation rather than at the end of an answer. That last part
is what multi-turn RL is supposed to buy: a policy that can trade a weaker move now (a small
concession to keep an impatient buyer at the table) for a better outcome later, instead of
optimizing each reply as if it were the last.

Eight turns of Craigslist haggling is a short horizon, but it is long enough for strategy to
show up in the transcripts (anchoring, concessions, holding or folding under a pass threat) and
short enough to run as a controlled experiment on one rented GPU. That makes it a small testbed
for a question bigger than the task: whether current training methods produce something like
persuasion, or just tighter answer-matching.

This repo is that experiment: three ways to teach a 4B model to bargain on the
CraigslistBargains seller task (**SFT**, **GRPO**, and **PPO**: same base model, same 817
training dialogues, same reward, same opponent, one shared eval harness, three seeds each). It
is the redesigned follow-up to
[llm-negotiation-rl](https://github.com/rgurney7/llm-negotiation-rl), which asked the same
question and failed at the finish line. Anything that must be identical across methods lives in
`shared/` and is imported by all three.

The headline table is below. The more interesting result is what the transcripts show: the
three methods learned three different strategies, but not necessarily negotiation.

## Results

Three seeds per method, evaluated greedily on 149 shared held-out scenarios against a buyer
model family never seen in training (`gpt-5.4-nano`). Full stats in
[results/aggregate.json](results/aggregate.json); every per-scenario transcript is in
`results/eval_s{1,2,3}/`.

| | mean reward (3-seed mean, range) | deal rate | price ratio |
|---|---|---|---|
| base (untrained) | 0.381 | 58% | 0.826 |
| **SFT** | **0.488** (0.459–0.506) | 86% | 0.788 |
| SFT, sanitized eval (see below) | 0.477 (0.464–0.487) | 95% | 0.758 |
| GRPO | 0.405 (0.396–0.415) | 99% | 0.722 |
| PPO | 0.427 (0.422–0.437) | 75% | 0.794 |

Every trained method beats the untrained base. Behaviour cloning beats both RL methods — SFT is
ahead of PPO on all three seeds (the contrast I picked as primary before the run) — and each RL
method distorts the deal-rate/price trade-off in exactly the direction its objective predicts:
GRPO closes almost every negotiation at the worst prices, PPO holds out for the best prices and
loses deals doing it. Three seeds is a descriptive result, not a significance claim; the ranges
above are ranges, not confidence intervals.

The whole run — nine training runs plus evaluation — took ~51 hours on one rented A40 and cost
about $23.

## What each method actually learned

The aggregate numbers hide the part worth reading the ~1,500 eval transcripts for: each method
has an unmistakable behavioral signature.

**SFT cloned the human surface.** Terse replies (8.75 words/turn on every seed; the human demo
median is 9), the human deal-rate/price trade-off, and one rare human phrase ("I can do $X",
1.9% of human seller turns) amplified into a template covering ~88% of its episodes. It wins by
not losing: its mean advantage over PPO comes from deals PPO fails to close, not from better
prices per deal. Per closed deal, they are even.

**GRPO learned an acceptance token, not a policy.** Trained on a single closing turn, two of
three seeds collapsed onto "That works for me." emitted regardless of context, including as the
answer to "What is the lowest you can sell the phone for?" It sometimes names a turn-1 price
and never defends it (all nine stated-then-broken floors were conceded on the very next turn),
and the buyer names the closing price first in ~97% of its deals. The 99% deal rate is real,
but the reward is the buyer's opening offer passed through: a policy that works only because
~98% of this buyer's openers already clear the reserve.

**PPO learned a concession ladder.** Highest anchor (~0.93x listing), the most distinct prices
per episode, descending "I can meet you at $X" counteroffers, slowest closes. But the steps do
not react to the buyer (correlation with buyer movement is ~0), and when the ladder runs out of
rungs it freezes: in ~98% of its lost deals the buyer's final offer was at or above the
reserve, a median ~0.31 reward left on the table each time.

**The untrained base reframes the comparison.** It is already the most fluent, most
item-grounded seller, the only one that argues from the listing, and the only one that reliably
moves the buyer up (~82% of its deals). It just fails to close (58%). Training did not teach
language or price extraction: the base had both. It taught closing, and every method bought
closing by discarding language. No trained method beats base on any grounding measure, and
grounding is uncorrelated with reward within every method.

One scenario shows all four signatures at once (a 2004 Volvo, $4,500 listing, buyer capped at
$3,150): base argues the service history, sets a $3,500 floor, holds it, and moves the buyer up
to $3,500, the only method that moved this buyer. SFT freezes at $3,900. GRPO opens "I'm open
to $4000, but that's my absolute floor," then accepts $3,150 on its next turn. PPO repeats "I
can meet you today at $4500." verbatim six times into a wall. (Scenario `0fbf2d4a…`, seed 1,
in `results/eval_s1/`.)

## The template leak, found and measured

Every SFT eval episode leaks its chat template: after the genuine reply, the model streams a
fake "user ... assistant" continuation of the dialogue that the live buyer reads, and in 21-33%
of SFT's deals per seed the agreed price first appears inside that hallucination. That put
SFT's headline number in doubt, so I re-ran the SFT eval with the leak truncated at generation
time (`eval_methods --sanitize-leak`; results in `results/eval_sanitized_s{1,2,3}/`). Sanitized
mean reward is 0.477 against 0.486 published, still ahead of both RL methods on every seed.
Deal rate rose from 85% to 95% and closing prices dropped about 3 points of listing: the
hallucinated continuations were costing SFT deals at roughly the rate their fake anchors were
earning price. The root cause (the fine-tune eroded the stop token even though it carried loss
in all 3,198 training examples) is still open, and blocks RL-from-SFT-init until fixed.

## What's being compared

A seller (LoRA adapter over `Qwen/Qwen3.5-4B`) negotiates against a buyer over a real
Craigslist listing. The reward is the agreed price, scaled to `[-1, 1]` against a reserve of
half the listing price (no deal → 0, below reserve → -1).

| | SFT | GRPO | PPO |
|---|---|---|---|
| Signal | behaviour clone | group-relative reward | actor–critic reward |
| Horizon | one supervised example per seller turn | single closing turn (1-step bandit) | multi-turn self-play (up to 8 turns, truncated at the deal) |
| Sees a live buyer at train time? | no (recorded transcript) | yes (one reply) | yes (every turn) |
| Objective | masked cross-entropy | token-level GRPO + k3 KL | token-level PPO, no KL, + critic |

The three horizons differ **by design** — each method runs in its standard, literature-faithful
form. Everything else is held identical (below). All three are then scored on the *same*
held-out scenarios through the *same* multi-turn eval.

PPO additionally **truncates each episode at the deal turn** the judge identifies: the terminal
reward lands on that turn and post-agreement turns are dropped before GAE, so PPO never trains on
chatter after the deal and the critic never has to value post-deal states. The cut is the *final*
settlement (a reopened-then-re-settled deal keeps the later price), so the reward is the price
actually obtained and caving is still penalized (`ppo/rollout.py`).

## Held constant

Enforced in code by `shared/` + `tests/test_render_parity.py`, which asserts the runtime
prompt is byte-identical to the committed data slices:

- **Base model + loader** — `Qwen/Qwen3.5-4B`, bf16, 16-bit LoRA, one loader for all three.
- **LoRA** — r=8, α=16, dropout=0, all-linear targets `q,k,v,o` + `gate,up,down`.
- **Prompt format** — system = seller prompt, user = `Negotiation Transcript: … [Your Turn]:`,
  byte-identical across SFT / GRPO / PPO / eval.
- **Reward** — `price_reward`, reserve = 0.5 × listing. An LLM judge (`gemini-3.1-flash-lite`)
  reads the finished transcript for deal + agreed price + close turn, with a stronger backup
  judge behind it; if both are unusable the sample is **dropped**, never scored — no fabricated
  no-deals, no guessed prices. Verdicts are cached by transcript hash, so the eval is replayable
  from `results/judge_cache.jsonl`. Generated text is sanitized of role-marker tokens in both
  transcript renders, so a generated utterance can never spoof a buyer turn in the text the
  reward source reads.
- **Opponent** — held-out design: train against `gemini-2.5-flash`, evaluate against a different
  family (`gpt-5.4-nano`), both anchored to the scenario's buyer target. A buyer-API failure
  drops the sample rather than scoring it as a walk-away.
- **Data + split** — the same 817 train dialogues (sliced per method), scenario-level de-leaked
  eval pool. Eval records persist each episode's full transcript alongside its score.
- **Seeds** — every source of run-to-run variance (data order, LoRA init, sampling, buyer draws)
  is seeded; the final tables are over seeds 1–3.

Per-method by necessity: the learning objective, the horizon, and the algorithm
hyperparameters. The policy LR is each algorithm's standard value — SFT 2e-4, GRPO 5e-5, PPO
1e-5 (PPO also trains a from-scratch critic at 3e-4) — so the comparison is of *recipes as
deployed*, not a single isolated knob: you can't give SFT and PPO the same LR without making one
of them non-standard. All three optimize with AdamW; schedules follow each method's standard
recipe. Eval decodes greedily for every method through one shared path.

## Layout

```
shared/      the single source of truth, imported by all three methods
  render.py        transcript + prompt formatting (matches the data pipeline byte-for-byte)
  reward.py        price_reward + deterministic extractor (offline; sanitizes markers)
  judge.py         LLM reward judge + backup + band gate, cached
  buyer.py         held-out API buyers (gemini train / gpt eval)
  env.py           NegotiationEnv: PPO self-play + GRPO single-close
  model.py         model loader + LoRA + chat-template prompt (one loader for all)
  data.py          slice loaders -> one Scenario shape
  eval_harness.py  the shared eval over eval_pool (paired win-rate + mean reward)
  persistence.py   checkpoint / HF mirror / teardown plumbing
  config.py        SharedConfig base; method configs subclass it
data/        the frozen canonical slices + the deterministic pipeline that built them
ppo/  grpo/  sft/   thin per-method packages: config + agent + loss/gae + train
results/     the N=3 run: aggregate stats, per-scenario eval transcripts, judge cache
tests/       offline tests (render parity, reward, gae, grpo-loss, masking, env/eval)
```

## Run

```bash
git clone https://github.com/rgurney7/negotiation-rl.git && cd negotiation-rl
pip install -e .                 # Python 3.11; training needs a CUDA box
cp .env.example .env             # GOOGLE_API_KEY (train buyer) + OPEN_AI_API_KEY (eval buyer)

python tests/run_all.py          # 127 offline checks, no GPU needed

python -m sft.train              # behaviour-clone baseline
python -m grpo.train             # group-relative RL
python -m ppo.train              # actor-critic RL
```

The multi-seed run is orchestrated by `run_final.py` (per-seed dirs and HF repos, resumable
per-unit, one shared eval at the end):

```bash
# on the GPU box — clone inside /workspace and set HF_HOME=/workspace/hf_cache so
# resume state survives a container restart:
python run_final.py --seeds 1 --methods ppo --eval-limit 150   # cheap de-risk probe
python run_final.py --seeds 1 2 3 --eval-limit 150             # the real thing

# on your own machine — the pod carries no cloud keys, so teardown runs locally:
caffeinate -i python local_killer.py --pod-id <id> --seeds 1 2 3
```

`--eval-limit 150` evaluates a seeded sample of 150 *distinct* listings, identical for every
method and seed. `aggregate_seeds.py` produces the headline table; at N<5 it reports descriptive
ranges and per-seed paired tests rather than pretending three seeds support a bootstrap CI.

## Artifacts

Everything the run produced is persisted (the previous iteration of this project lost its
checkpoints to ephemeral compute; this one over-corrects):

- **In this repo:** `results/` — aggregate stats, all per-scenario eval transcripts for every
  method × seed, the sanitized SFT re-eval, and the judge cache that makes the scoring
  replayable.
- **On Hugging Face:** per-seed adapters + training logs + rollout transcripts at
  `ShallowLearning/negotiation-{sft,grpo,ppo}-qwen3.5-4b-s{1,2,3}`, and the results mirror at
  `ShallowLearning/negotiation-results`.

## Relation to the first attempt

This is the second iteration of
[llm-negotiation-rl](https://github.com/rgurney7/llm-negotiation-rl). The first version produced
real training-time findings, but its final comparison had flaws I documented at the time and
couldn't repair after the fact: the committed eval read the training file, GRPO and PPO ran in
different environments, everything was single-seed, and the checkpoints weren't persisted. This
repo asks the same question as a controlled experiment.

## License

MIT.
