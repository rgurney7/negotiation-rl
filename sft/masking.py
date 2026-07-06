"""Tokenization + loss masking for seller-turn SFT examples."""


def _merge_system_into_user(messages):
    """Fold a leading system message into the first user turn (templates with no system role)."""
    out, pending = [], None
    for m in messages:
        if m["role"] == "system":
            pending = m["content"]
            continue
        if pending is not None and m["role"] == "user":
            m = {"role": "user", "content": pending + "\n\n" + m["content"]}
            pending = None
        out.append(m)
    if pending is not None:
        out.insert(0, {"role": "user", "content": pending})
    return out


def _render(messages, tokenizer, add_generation_prompt):
    """apply_chat_template, trying (orig vs system-merged) x (enable_thinking=False vs plain),
    first that works."""
    for msgs in (messages, _merge_system_into_user(messages)):
        for extra in ({"enable_thinking": False}, {}):
            try:
                return tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=add_generation_prompt, **extra)
            except Exception:
                continue
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt)


def build_labels(messages, tokenizer, max_seq_len):
    """One chat example -> {input_ids, labels, attention_mask}; loss on assistant tokens only."""
    from shared.model import build_prompt
    sys_content = next((m["content"] for m in messages[:-1] if m["role"] == "system"), "")
    usr_content = next((m["content"] for m in messages[:-1] if m["role"] == "user"), "")
    prompt = build_prompt(tokenizer, sys_content, usr_content)
    full = _render(messages, tokenizer, add_generation_prompt=False)

    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    full_ids = tokenizer(full, add_special_tokens=False).input_ids

    n_prompt = len(prompt_ids)
    if full_ids[:n_prompt] != prompt_ids:        # template tokenized across the boundary
        n_prompt = 0
        for a, b in zip(prompt_ids, full_ids):
            if a != b:
                break
            n_prompt += 1

    full_ids = full_ids[:max_seq_len]
    n_prompt = min(n_prompt, len(full_ids))
    labels = [-100] * n_prompt + full_ids[n_prompt:]
    return {"input_ids": full_ids, "labels": labels, "attention_mask": [1] * len(full_ids)}


def build_dataset(examples, tokenizer, max_seq_len):
    """HF Dataset of {input_ids, labels, attention_mask}; drops fully-masked (over-long) rows."""
    from datasets import Dataset
    rows = [build_labels(e["messages"], tokenizer, max_seq_len) for e in examples]
    kept = [r for r in rows if any(t != -100 for t in r["labels"])]
    dropped = len(rows) - len(kept)
    if dropped:
        print(f"  WARN: dropped {dropped}/{len(rows)} fully-masked (over-long) examples", flush=True)
    return Dataset.from_list(kept)


def inspect_example(examples, tokenizer, max_seq_len, k=0):
    """Decode one example, showing exactly which tokens carry loss (manual masking check)."""
    row = build_labels(examples[k]["messages"], tokenizer, max_seq_len)
    masked = [t for t, l in zip(row["input_ids"], row["labels"]) if l == -100]
    trained = [t for t, l in zip(row["input_ids"], row["labels"]) if l != -100]
    print("=" * 70 + "\nMASKED (no loss) — system + transcript + buyer turns:\n")
    print(tokenizer.decode(masked))
    print("\n" + "=" * 70 + "\nTRAINED (loss) — seller target + stop token:\n")
    print(repr(tokenizer.decode(trained)))
    print("=" * 70 + f"\n{len(masked)} masked, {len(trained)} trained tokens")
    print("Seller target was:", repr(examples[k]["messages"][-1]["content"]))
