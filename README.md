# negotiation-rl

A controlled comparison of three ways to teach a 4B language model to bargain: **SFT**,
**GRPO**, and **PPO**, all on the same CraigslistBargains seller task, the same base model, the
same data, the same opponent, and the same reward — trained as identically as the algorithms
allow, and evaluated through one shared harness.

The point is not to crown a winner in the abstract. It is to hold every experimental condition
constant so that whatever difference appears between the three is attributable to the *learning
algorithm*, not to a confound. The repository is organized around that single idea: anything
that must be identical across methods lives in `shared/` and is imported by all three.

## Results

Three seeds per method, evaluated greedily on 149 shared held-out scenarios against a buyer
model family never seen in training (`gpt-5.4-nano`). Full stats in
[results/aggregate.json](results/aggregate.json); every per-scenario transcript is in
`results/eval_s{1,2,3}/`.

| | mean reward (3-seed mean, range) | deal rate | price ratio |
|---|---|---|---|
| base (untrained) | 0.381 | 58% | 0.826 |
| **SFT** | **0.488** (0.459–0.506) | 86% | 0.788 |
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
  method × seed, and the judge cache that makes the scoring replayable.
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
