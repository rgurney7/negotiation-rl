"""Deterministic build: LLM enrichment cache + pinned HF data -> training slices, eval pools, manifest."""
import hashlib
import json
import os
import random

import enrich_common as ec
from enrich_llm import CACHE_PATH

HERE = os.path.dirname(__file__)
OUT = HERE
SLICES = os.path.join(HERE, "slices")
VAL50_N = 50
VAL50_SEED = 0
MISMATCH_REL = 0.05  # drop if |llm_close - acts_agreed| > max($2, 5% of listing)


def load_cache():
    done = {}
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done[rec["did"]] = rec
    return done


def resolve_close(d, rec):
    """Pass B's close if it's a contentful seller turn, else fall back; returns (close_idx, used_fallback)."""
    ci = rec.get("close_turn_index", -1)
    by_idx = {t.idx: t for t in d.turns}
    t = by_idx.get(ci)
    if t is not None and t.role == "seller" and not t.is_empty:
        return ci, False
    return ec.last_contentful_seller_index(d), True


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def main():
    cache = load_cache()
    if not cache:
        raise SystemExit(
            f"LLM enrichment cache empty/missing at {CACHE_PATH}. Run `python enrich_llm.py` first, "
            "or use the frozen tables in slices/ (the canonical build). Refusing to emit empty tables.")
    ds = ec.load_hf()

    # --- scenario keys present in eval splits (for de-leak) ---
    val_dialogues = list(ec.iter_dialogues(ds, "validation"))
    test_dialogues = list(ec.iter_dialogues(ds, "test"))
    eval_sids = {d.sid for d in val_dialogues} | {d.sid for d in test_dialogues}

    # --- walk golden train, apply drops, build slices ---
    counts = dict(train_dialogues=0, golden=0, missing_llm=0, leak=0, concession=0,
                  passb_no_deal=0, close_unreliable=0, orphan=0, kept=0, close_fallback=0,
                  sft_opener_dropped=0)
    dropped = []
    enriched_rows, sft_rows, grpo_rows, ppo_rows = [], [], [], []

    for d in ec.iter_dialogues(ds, "train"):
        counts["train_dialogues"] += 1
        ok, _, outcome = ec.golden_gate(d)
        if not ok:
            continue
        counts["golden"] += 1

        rec = cache.get(d.did)
        if rec is None:
            counts["missing_llm"] += 1
            dropped.append({"did": d.did, "reason": "missing_llm"})
            continue
        if d.sid in eval_sids:
            counts["leak"] += 1
            dropped.append({"did": d.did, "reason": "leak"})
            continue
        if rec["concessions"]:
            counts["concession"] += 1
            dropped.append({"did": d.did, "reason": "concession"})
            continue

        agreed = outcome["agreed"]
        # acts decide deal-existence; Pass B decides which turn closes it
        if rec["close_type"] == "no_deal":
            counts["passb_no_deal"] += 1
            dropped.append({"did": d.did, "reason": "passb_no_deal"})
            continue
        llm = rec["llm_agreed_price"]
        if llm is not None:
            ratio = llm / d.listing
            material = abs(llm - agreed) > max(2.0, MISMATCH_REL * d.listing)
            if not (ec.RATIO_BAND[0] <= ratio <= ec.RATIO_BAND[1]) or material:
                counts["close_unreliable"] += 1
                dropped.append({"did": d.did, "reason": "close_unreliable"})
                continue

        close, fb_used = resolve_close(d, rec)
        if close is None:
            dropped.append({"did": d.did, "reason": "no_close"})
            continue
        if fb_used:
            counts["close_fallback"] += 1

        # orphan guard: an accept-close must have the agreed price already in context
        # (a proposes-close introduces it at the close)
        if rec["close_type"] == "seller_accepts" and not ec.price_in_context(d, close, agreed):
            counts["orphan"] += 1
            dropped.append({"did": d.did, "reason": "orphan"})
            continue

        counts["kept"] += 1
        sys_prompt = ec.seller_prompt(d)
        by_idx = {t.idx: t for t in d.turns}
        fb = ec.first_buyer_msg_index(d)

        enriched_rows.append({
            "did": d.did, "sid": d.sid, "split": "train",
            "listing": d.listing, "buyer_target": d.buyer_target, "seller_target": d.seller_target,
            "reserve": d.reserve, "agreed_price": agreed, "closing_ratio": agreed / d.listing,
            "category": d.category, "title": d.title,
            "description": ec.item_description(d),
            "close_turn_index": close, "close_type": rec["close_type"],
            "first_buyer_msg_index": ec.first_buyer_msg_index(d),
            "seller_prompt": sys_prompt,
            "turns": [t.as_dict() for t in d.turns],
        })

        # SFT: one pair per contentful seller turn at/before the close, skipping
        # seller turns before the first buyer message (off-distribution at inference)
        for t in d.turns:
            if t.role == "seller" and not t.is_empty and t.idx <= close:
                if fb is not None and t.idx < fb:
                    counts["sft_opener_dropped"] += 1
                    continue
                sft_rows.append({
                    "did": d.did, "sid": d.sid, "turn_index": t.idx, "is_close": (t.idx == close),
                    "system": sys_prompt, "user": ec.render_context(d, t.idx), "assistant": t.text,
                })

        # GRPO: context truncated to before the close; policy regenerates it
        grpo_rows.append({
            "did": d.did, "sid": d.sid, "system": sys_prompt,
            "user": ec.render_context(d, close),
            "close_turn_index": close, "close_type": rec["close_type"],
            "reference_close": by_idx[close].text,
            "listing": d.listing, "reserve": d.reserve, "buyer_target": d.buyer_target,
            "title": d.title, "description": ec.item_description(d),
        })

        # PPO: buyer-opener-only seed so the policy self-plays every seller turn, matching eval
        ppo_rows.append({
            "did": d.did, "sid": d.sid, "system": sys_prompt,
            # cold open (fb is None): render like the eval pool below
            "user": ec.render_context(d, fb + 1, from_idx=fb) if fb is not None
            else ec.render_context(d, 0),
            "seed_through_index": fb,
            "listing": d.listing, "reserve": d.reserve, "buyer_target": d.buyer_target,
            "title": d.title, "description": ec.item_description(d),
        })

    # --- eval pools: distinct scenarios from val/test (NOT filtered) ---
    def eval_pool(dialogues, split, exclude_sids=frozenset()):
        seen, rows = set(), []
        for d in dialogues:
            if d.sid in seen or d.sid in exclude_sids:
                continue
            seen.add(d.sid)
            fb = ec.first_buyer_msg_index(d)
            rows.append({
                "sid": d.sid, "split": split, "listing": d.listing,
                "buyer_target": d.buyer_target, "seller_target": d.seller_target,
                "reserve": d.reserve, "category": d.category, "title": d.title,
                "description": ec.item_description(d),
                "seller_prompt": ec.seller_prompt(d),
                # buyer-opener-only seed; fb is None -> model opens cold
                "first_buyer_msg": ec.render_context(d, fb + 1, from_idx=fb) if fb is not None
                else ec.render_context(d, 0),
                "first_buyer_index": fb,
            })
        return rows

    val_pool = eval_pool(val_dialogues, "validation")
    # de-leak: a scenario in both splits stays in val, dropped from test
    val_sids = {r["sid"] for r in val_pool}
    test_pool = eval_pool(test_dialogues, "test", exclude_sids=val_sids)
    val50 = random.Random(VAL50_SEED).sample(val_pool, min(VAL50_N, len(val_pool)))

    # --- write everything ---
    paths = {
        "enriched_train.jsonl": enriched_rows,
        "eval_pool.jsonl": val_pool + test_pool,
        os.path.join("slices", "sft.jsonl"): sft_rows,
        os.path.join("slices", "grpo.jsonl"): grpo_rows,
        os.path.join("slices", "ppo.jsonl"): ppo_rows,
        os.path.join("slices", "val50.jsonl"): val50,
    }
    shas = {}
    for rel, rows in paths.items():
        p = os.path.join(OUT, rel)
        write_jsonl(p, rows)
        shas[rel] = sha256(p)

    counts["sft_pairs"] = len(sft_rows)
    counts["grpo_examples"] = len(grpo_rows)
    counts["ppo_examples"] = len(ppo_rows)
    counts["val_scenarios"] = len(val_pool)
    counts["test_scenarios"] = len(test_pool)
    counts["val50"] = len(val50)

    manifest = {
        "hf_repo": ec.HF_REPO, "hf_revision": ec.HF_REVISION,
        "model": "gemini-3.1-flash-lite-preview", "temperature": 0,
        "gate": {"ratio_band": ec.RATIO_BAND, "price_band": ec.PRICE_BAND,
                 "min_contentful_turns": ec.MIN_CONTENTFUL_TURNS,
                 "min_seller_priced": ec.MIN_SELLER_PRICED, "min_buyer_priced": ec.MIN_BUYER_PRICED},
        "counts": counts, "output_sha256": shas,
        "dropped_ids": dropped,
    }
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print("=== ENRICH BUILD COMPLETE ===")
    for k, v in counts.items():
        print(f"  {k:18s}: {v}")
    print(f"  outputs: {', '.join(paths.keys())}, manifest.json")


if __name__ == "__main__":
    main()
