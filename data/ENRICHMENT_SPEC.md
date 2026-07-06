# Enrichment spec — building the SFT/GRPO/PPO training + eval data

Source of truth for the data pipeline that turns raw CraigslistBargains into the
canonical training tables for the fair SFT/GRPO/PPO comparison. Last updated 2026-06-27.

## 0. Goal & invariants
- Produce ONE quality/golden-filtered **training** pool, sliced three ways (SFT / GRPO / PPO)
  so all three train on the **same examples** — only the slice and the learning signal differ.
- All three see a **byte-identical context format**. Eval is identical and seeded across methods.
- Golden filtering applies to **training only**. Val/test are never golden/concession-filtered.
- Deterministic wherever possible; the only model-in-the-loop steps are two narrow LLM passes
  (`enrich_llm.py`), whose outputs are cached upstream. The **frozen `slices/` tables are the
  reproducible build output**; a from-scratch rebuild from raw HF needs that Pass-A/B cache present
  first (`enrich.py` now hard-fails on an empty/missing cache rather than emitting empty tables).

## 1. Data source — HuggingFace spine (no CodaLab)
- Spine = `stanfordnlp/craigslist_bargains` (HF), **pinned to a fixed revision** recorded in the
  manifest. Load with `trust_remote_code=True` behind the SSL shim (see scripts).
- Why HF-only: HF carries per-turn `dialogue_acts` (intent + price), `agent_info` (Role/Target/
  Bottomline), `items` (Title/Description/Category/Price), `utterance`, `agent_turn`. That is
  everything the gate, the slices, and the prompts need. HF *train* acts are accurate (validated
  100% vs CodaLab on tagged deals). CodaLab is NOT used: its only unique value was `final_price`,
  which **eval does not need** (eval scores the model's OWN transcript, not the human one), and a
  `uuid` join is impossible anyway (HF exposes no id; HF splits ⊋ CodaLab splits).
- Deliberate tradeoff: deals with no `accept` act (untagged) are excluded. They lack a clean close,
  so they fail golden-quality regardless. Net golden is LARGER than the CodaLab path (~2.1k vs 1.8k).
- HF *test* acts are broken (0 tagged deals) — irrelevant here, because val/test need only the
  **scenario** (listing, targets, item), never the acts.

## 2. Identifiers
- `dialogue_id = sha1(normalized full transcript)` — stable across HF reorderings; doubles as a
  dedup key and the LLM-cache key.
- `scenario_id = sha1(listing_price | buyer_target | item_title | item_description)` — groups the
  ~1.6 dialogues that share a scenario; used for split de-leaking and paired eval.

## 3. Turn model & cleaning
Each raw HF turn → `{idx, role, intent, price, text, is_empty, is_action_marker, offers_addon}`.
- `role` from `agent_info.Role[agent_turn[idx]]`; `intent`/`price` from `dialogue_acts`.
- **Empty-marker drop (deterministic, no LLM, no text-regex):** drop every turn with
  `text.strip()==""`. Proven structural fact on HF train: the four action intents
  `{offer, accept, reject, quit}` are **100% empty (7,606/7,606)**; every dialogue intent is 0% empty.
  - Rationale beyond emptiness: these are AMT **UI button-clicks** (offer/accept/reject/quit). Our
    environment has no buttons — the seller types, price is extracted from text, reward scores the
    text. Keeping them would teach a mechanic absent at inference. Coherence checked: ~94.5% of
    markers are post-close (never in any training context); offer-marker "orphan" rate (price not in
    prior contentful text) ≈ 2% and mostly `$7`-vs-`7` artifacts. Stripped dialogues read naturally.
  - **Guard:** assert the agreed price appears in the contentful context of every SFT target and
    GRPO cut; flag/drop the rare (<2%) violation.
- `intent` is retained as metadata (labels the contentful acts; corroborates Pass B's close).

## 4. Deterministic golden gate (TRAIN only)
`golden = deal_reached & 0.5 ≤ agreed/listing ≤ 1.5 & num_turns ≥ 4 & seller_priced ≥ 2 &
buyer_priced ≥ 1 & no-empty/garbage`.
- `deal_reached` = an `accept` act exists; `agreed` = last priced `offer` before it (acts-derived).
- `seller_priced/buyer_priced` = count of that side's turns carrying a price (act price ≠ −1).
- Price floor **0.5×listing is reward-aligned** (the reward reserve; below it → −1). Single knob;
  raise to 0.6–0.7 for stronger demos. `u_seller` is a REPORTING metric, never a filter.
- Yields ~2,138 golden train dialogues (acts-tagged).

## 5. LLM enrichment — two passes (`enrich_llm.py`, cached upstream)
Model `gemini-3.1-flash-lite-preview`, `temp=0`, Pydantic structured output, few-shot prompts.
Run on golden candidates only; ~2 calls/dialogue (parallelizable); ~$5 one-time. Cache keyed by
`dialogue_id` → skip-if-present. The cache is the upstream artifact behind the frozen `slices/`;
`enrich.py` hard-fails if it is empty/missing rather than silently dropping every dialogue.
- **Pass A — concession detection.** Flags SELLER turns offering value beyond the item at the price
  (delivery, bundles/throw-ins, freebies, warranties, free utilities). Ignores neutral logistics
  ("meet you somewhere"), price-drops, and BUYER concessions. Output: per-turn `offers_addon`.
  Validated 16/16 on curated hard cases.
- **Pass B — closing turn.** Returns the SELLER turn that locks the FINAL price (propose or
  accept-in-words), never an empty marker, the LATER turn on reopen. Output: `close_turn_index,
  close_type, llm_agreed_price, renegotiated_after_first_agreement`. Validated 16/16 incl.
  accept-with-no-number-in-turn and multi-anchor (33→29, 14k→13k).
- **Cross-check + close-reliability drop:** `llm_agreed_price` vs acts-derived `agreed`. The acts
  gate is authoritative on deal-existence, but Pass B must corroborate WHICH turn closes. Drop the
  dialogue when Pass B returns `no_deal`, or its close price is out of the [0.5,1.5] ratio band, or
  differs from the acts agreed by > max($2, 5% of listing) — these are acts false-positive deals or
  wrong-turn closes that would corrupt the GRPO cut. (Validated: e.g. a buyer who walks at $400 while
  the seller holds $550 was acts-tagged a $400 deal; Pass B correctly flags no_deal.)
  Caveat (second audit): the ratio/materiality corroboration is gated on `if llm_agreed_price is not
  None`, so a deal-typed close with a **null** Pass-B price is kept uncorroborated. This affects only
  the close-turn *index* (the context-truncation point) — the reward target `agreed_price` is
  acts-derived and already band-validated, and `close_type` is never read by training. Bounded
  (`close_unreliable` drops = 7 in the frozen build). Left as-is: the slices are frozen, so tightening
  the gate without regenerating would desync code from data.

## 6. Concession drop (shared pool — all three)
DROP any dialogue with ≥1 seller concession from the **shared** pool (not mask, not SFT-only —
preserves the same-examples invariant). ~26% of golden (regex; LLM may push to ~30–35%).
Survivors ≈ 1,311–1,580 dialogues / ~1,182 scenarios / ~6k SFT pairs — sufficient (SFT plenty for
4B LoRA; RL is compute-bound, not scenario-bound). Unbiased: concession vs clean have identical
closing ratio (0.81/0.81) and buyer difficulty (0.70/0.70); only a mild category reweight. The
bundle bought no extra price → those are value-worse closes; dropping costs zero price signal and
removes reward-gaming demos.

## 7. The three slices (identical context encoding)
`system = seller_prompt`; `user = "Negotiation Transcript:\n[Buyer]: …\n[Seller]: …\n\n[Your Turn]:"`;
`assistant = the seller turn`. Header is **"Negotiation Transcript:"**. `seller_prompt` is built from
item fields:
```
You are a seller on Craigslist. Your goal is to maximize the sale price while still closing the deal.
You listed this item at ${listing}.

Item: {title}
Description: {description}

Negotiate on price only. Do not offer extras, add-ons, free items, delivery, warranties, or
anything beyond the item itself.

Write your next message only. One to three sentences of natural dialogue. Do not start your
message with any label or prefix. Do not write the buyer's response.
```
- **SFT** — per-turn `(context → next seller turn)` examples (NOT role-alternating chat, so the
  tokenization matches the RL inputs). `sft_eligible` seller turn = `not is_empty AND
  idx ≤ close_turn_index AND idx ≥ first_buyer_msg_index AND not offers_addon`. The `≥ first_buyer`
  clause drops the seller's greeting-into-an-empty-transcript when the seller opens (~44% of
  dialogues) — off-distribution, since at inference the seller always responds to a buyer; the opener
  is still kept as CONTEXT for later turns. Loss masked to the assistant turn.
- **GRPO** — context truncated to **before** `close_turn_index`; the policy regenerates the close.
  One buyer reply, then judge. Carries reward meta (`listing`, `reserve=0.5·listing`).
- **PPO** — scenario only; transcript seeded with the **real human prefix through the first buyer
  message** (`first_buyer_msg_index`); the policy owns every seller turn after. Full self-play vs
  the seeded LLM buyer to the horizon; reward at horizon. (Buyer opens 56% of the time; when the
  seller opens, seed through the first buyer message so the policy always responds to a real buyer.)

## 8. Eval pools (val/test — NOT filtered)
- Built deterministically from HF val/test scenarios (no LLM passes). Each row: `scenario_id`,
  item fields, `listing/buyer_target/seller_target`, `seller_prompt`, real opening message.
- Materialize a **seeded ~50-scenario val subset** (`val50`) for intermittent checkpoint/HP
  selection; full test run once. All methods get identical eval scenarios + buyer realizations
  (fixed seed). Pair by `scenario_id`. Primary metric = mean reward/utility over scenarios
  (no-deal = floor) + paired win-rate vs SFT; ≥3 seeds.

## 9. Splits & de-leak
- Native HF train/val/test, keyed by `scenario_id`. Compute scenario keys across all splits; **drop
  train dialogues whose `scenario_id` appears in val/test.** `check_splits` asserts zero overlap.

## 10. Outputs (`enrich.py`) & reproducibility
- `cache/llm_enrichment.jsonl` — Pass A/B raw outputs (upstream artifact; the frozen `slices/` are
  the reproducible output, and `enrich.py` hard-fails if this cache is empty/missing).
- `enriched_train.parquet` — canonical golden-train rows (all fields incl. `turns[]`, flags,
  `close_turn_index`, `first_buyer_msg_index`, `seller_prompt`).
- `eval_pool.parquet` — val/test scenarios.
- Slices: `sft.jsonl`, `grpo.jsonl`, `ppo.jsonl`, `val50.jsonl`.
- `manifest.json` — HF revision, model id, temp, per-stage counts, SHA-256 of each output, and the
  dropped-id list with reasons (`gate` / `concession` / `leak` / `orphan`).

## 11. Verification checklist (run before declaring done)
- Scenario de-leak asserts zero train↔val/test overlap.
- Acts vs LLM agreed-price agreement rate reported; mismatches eyeballed.
- Re-render 5 SFT pairs + 5 GRPO contexts + 5 PPO seeds; read end-to-end for coherence.
- No concession dialogue survives; no empty marker is ever an SFT target; agreed price present in
  every SFT/GRPO context (orphan guard).
- Final counts match plan (golden ~2.1k → ~1.3–1.6k post-drop); SFT pair count; val/test scenarios.

## 11b. Build results (2026-06-28, paid key, full run — all assertions pass)
Funnel: 5,138 train dialogues -> 1,385 golden (strict gate) -> drops: 531 concession (38%, LLM,
precision-checked), 26 Pass-B no_deal, 7 close-unreliable, 3 orphan, 1 leak -> **817 kept**.
Outputs: 3,198 SFT pairs (after dropping 362 pre-buyer opener targets) · 817 GRPO · 817 PPO ·
401 val + 508 test eval scenarios · 50 val50.
Verified: de-leak overlap 0 · 0 kept concessions/no_deals · 0 empty SFT targets · GRPO user ==
SFT@close user for all 817 (byte-identity) · close present in SFT 817/817 · all closing ratios in
[0.5,1.5] (mean 0.807). Concession rate came in at 38% (LLM) vs 26% (regex) — the LLM is more
thorough (utilities, non-price conditions, "bring it to you"); sampled flags were 100% true-positive.
LLM pass: gemini-3.1-flash-lite-preview, temp=0, 1,385 dialogues x2, 0 failures, ~$1.
VOLUME LEVER: `MIN_SELLER_PRICED = 1` in enrich_common.py recovers single-anchor deals
(~2,000 golden -> ~1,200 kept) if more headroom is wanted; 817 is already sufficient (SFT plenty for
LoRA, RL compute-bound).

## 11c. Pre-hookup data audit (2026-06-28) — findings, decisions, hookup flags
Full audit of the produced slices (template/alignment/leakage/headroom/degeneracy). Verified clean:
all 4 slices cover the same 817 dids · GRPO.user==SFT@close.user 817/817 · PPO seed ends on a
`[Buyer]` line 817/817 (never bare header) · system prompt identical across slices · 0 target
leakage (`[Buyer]`/`[Seller]`/`[Your Turn]` never in an SFT target) · SFT contexts are nested
prefixes · 0 duplicate (sys,user,asst) triples · prompt sizes tiny (~720 tok GRPO, ~320 PPO).

FIXED — **val/test scenario leak**: 2 `scenario_id`s (the same $3,395 duplex) appeared in BOTH HF
val and test pools. `enrich.py` now de-leaks test against val (`test -= val_sids`); **test 508 -> 506**,
val unchanged at 401, val50 unaffected. Only `eval_pool.jsonl` SHA changed; all train slices
byte-identical. Re-verified val/test overlap = 0, train↔eval leak = 0.

DECISION — **GRPO keeps the single-final-turn cut** (user, 2026-06-28). 338/817 (41%) closes are
`seller_accepts`, where the buyer already named the in-band price (verified: agreed price is in
context for all 338) so the policy's final-turn decision is low-variance (accept vs. squeeze). This
makes GRPO a 1-turn bandit vs. PPO's full self-play — a deliberate, documented HORIZON ASYMMETRY,
accepted to preserve the byte-identical "same 817 samples" invariant. (Rejected alt: an earlier cut
for multi-turn GRPO — would break GRPO.user==SFT@close.user.) `close_type` mix: 479 seller_proposes,
338 seller_accepts.

DECISION — **eval/test seed = BUYER-OPENER-ONLY** (user, 2026-06-28). The eval `first_buyer_msg` now
renders ONLY the buyer's first message (`render_context(d, fb+1, from_idx=fb)`), dropping any human
seller opening preamble, so the policy generates EVERY seller turn (incl. the opening) against the
simulated buyer; the buyer's first human message is kept as a fixed anchor for cross-method
comparability. Verified: 0/907 eval seeds contain a `[Seller]:` line (was 385/907); 891 seeds = one
`[Buyer]` opener; 16 cold-open (no buyer msg -> empty seed, model opens cold).
APPLIED to PPO too (user, 2026-06-28): the PPO *training* seed (`slices/ppo.jsonl`) is now also
buyer-opener-only (`from_idx=fb`), so PPO train ≡ eval (policy self-plays every seller turn incl. the
opening). Verified: 0/817 PPO seeds contain `[Seller]:` (was 362). Context sufficiency checked: the
system prompt always carries the full scenario; 455/817 (56%) dialogues were already buyer-open;
dropped seller openers are 85% greetings (54/362 contain a digit). Correction (second audit): of
those 54, 31 restate the listing (already in the system prompt) but **11 carry a seller price anchor
that DIFFERS from the listing** (e.g. listing $175 / opener "$200") — a genuine negotiation signal
that SFT/GRPO context retains and PPO/eval never see. Bounded (~1.3% of train rows) but not strictly
contentless, so disclosed as an SFT/GRPO-vs-PPO input asymmetry (see FINDINGS "Disclosed
asymmetries"). Resulting buyer seeds median 11 words. Tradeoff: PPO-seed no longer == SFT@first for
the 362 seller-open dialogues (it now matches the EVAL convention instead) — GRPO==SFT@close unchanged.
Net file deltas across both seed changes: `enriched_train/sft/grpo` byte-identical; `eval_pool`,
`val50`, `ppo` changed.

FLAGS for the SFT/GRPO/PPO hookup agents (data facts they must honor):
1. **Reward parse fallback** — 21% of closes (incl. ~all 338 seller_accepts) restate NO number
   ("Okay. That works."). A reward extractor reading only the seller's final message will fail to
   score them; it MUST fall back to the buyer's last in-band offer in context.
2. **Reward must be RATIO-based** — listings span $5–$53,000 (10,600×). Confirm `shared/reward.py` is
   `price_reward(agreed, reserve=0.5·listing, ceiling=listing)` (= 2·ratio−1, scale-free). An
   absolute-dollar reward would wreck advantage normalization across that spread.
3. **Headroom is healthy, not saturated** — human demos: median closing ratio 0.82 → implied reward
   ~0.65 (mean 0.61); only 33/817 near ceiling, 6 at/below reserve. SFT imitates ~0.65-reward play,
   so RL has real room to beat it (the point of the comparison).
4. **16 eval scenarios (1.8%)** have no contentful buyer opener (`first_buyer_index=None`, empty
   seed) — LEFT IN by choice; the eval harness should let the LLM buyer open rather than seed empty.
5. **53/817 (6.5%)** scenarios have `buyer_target` < reserve (0.5·listing): a PPO buyer emulating that
   human anchors below the positive-reward zone (hard, not a bug). Train covers 767 distinct
   scenarios across 817 dialogues (44 repeat, max 4×).

REPRODUCIBILITY CAVEAT: newer `datasets` rejects the loading-script `trust_remote_code` and ignores
the pinned revision, falling back to the **locally cached** copy. The cache matches the original
pinned build (identical train counts/dids/SHAs), so current artifacts are faithful — but a fresh
machine with a different `datasets` version could diverge. Re-run uses `/Users/RyanG/RL/.venv`.

## 11d. Buyer / opponent design (2026-06-28)
The buyer is the AI that plays the customer; it negotiates against the seller policy during RL training
and during grading. SFT never touches it. Decisions:
- **NO persona — role-from-context.** Buyer prompt = the listing (title + description + price) + its
  private target (`buyer_target`); behavior emerges from context, no hand-authored character. Needs the
  item `description`, which is why a standalone `description` field was added to the buyer-facing rows
  (see below).
- **HELD-OUT buyer (user, 2026-06-28).** Train all RL methods against buyer **A**; grade everyone
  against a different, capability-matched, different-lineage buyer **B**. Rationale: PPO (online RL,
  high buyer exposure) can overfit one opponent's idiosyncrasies; grading on a fresh B measures
  transferable skill and protects the ranking. GRPO exposure is low (one reply/example), SFT zero.
  Recommend ALSO grading vs A so the (A−B) gap per method directly measures overfitting.
- **Models:** A (train) = `gemini-2.5-flash`, B (grade) = `gpt-5.4-nano`. Gemini is faster (~0.6s vs
  ~1.3s) and is the in-loop buyer; OpenAI as grader keeps the grade independent of the Gemini-family
  data-enrichment model. Both AMERICAN-made, different lineage.
- **Reasoning OFF + standardized (verified 2026-06-28):** `gemini-2.5-flash` →
  `thinking_config.thinking_budget=0` (thoughts=None); `gpt-5.4-nano` → `reasoning_effort="none"`
  (reasoning_tokens=0; it rejects "minimal", scale is none/low/medium/high/xhigh). Both produced clean
  target-anchored buyer replies with zero reasoning tokens. Keys in `.env` (`GOOGLE_API_KEY`,
  `OPEN_AI_API_KEY`), both validated.
- **Throughput (hookup flag):** both are API models; the train buyer runs in the PPO rollout loop, so
  fire buyer calls CONCURRENTLY across the vectorized envs (sequential would crawl). Watch RPM/TPM at
  high concurrency; use fixed `seed`+temperature on the grader for reproducible eval (≥3 seeds). Cost
  modest (nano/flash). Caveat: A and B are strong frontier-small models, likely stronger than the 4B
  seller — a tough but consistent adversary; comparison stays valid (all methods vs the same grader).
- **Opening:** eval & PPO-train both start from the human buyer opener anchor; buyer A/B plays every
  subsequent buyer turn (16 cold-open scenarios have no opener → buyer opens or seller opens cold).

DATA CHANGE — standalone `description` field added (2026-06-28) to `enriched_train`, `eval_pool`,
`slices/ppo`, `slices/grpo` (+ `title` on ppo/grpo) so the buyer can read the same listing the seller
sees (verified verbatim-identical to the seller-prompt description on 907/907 eval rows → no info edge).
`slices/sft.jsonl` deliberately UNCHANGED (SFT never uses the buyer) — byte-identical. Re-verified
counts unchanged (817 kept, 3198 SFT, 401 val, 506 test).

## 12. File architecture
- `data/enrich_common.py` — HF load (pinned), `dialogue_id`/`scenario_id`, turn cleaning/rendering,
  golden-gate predicate, `seller_prompt` builder.
- `data/enrich_llm.py` — Pass A + Pass B with caching → `cache/llm_enrichment.jsonl`.
- `data/enrich.py` — deterministic build: gate → concession drop → slices → eval pools → de-leak →
  artifacts + manifest.

## 13. Open / out of scope
- Init (SFT-init vs base-independent) — separate decision, gated on a base-model `extract_price`
  hit-rate check.
- Optional Stage-2 LLM coherence gate — only if a spot-check shows junk surviving.
- "Does RL benefit from more/harder data?" — parked ablation (RL on the fuller pool).
