import argparse
import os

from transformers import Trainer, TrainingArguments, TrainerCallback, DataCollatorForSeq2Seq

from shared import data, model as model_mod, persistence
from shared.seeding import seed_everything
from .config import SFTConfig
from .masking import build_dataset, inspect_example


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true", help="print one masked example and exit")
    ap.add_argument("--max-examples", type=int, default=None, help="cap examples (smoke test)")
    ap.add_argument("--seed", type=int, default=None, help="override cfg.seed (for multi-seed runs)")
    ap.add_argument("--preview", action="store_true",
                    help="tiny smoke run (default 50 examples), local-only: no HF push, no pod stop, "
                         "isolated *_preview run dir.")
    args = ap.parse_args()

    cfg = SFTConfig()
    if args.preview:
        args.max_examples = args.max_examples or 50
        persistence.mark_preview(cfg)
    if args.seed is not None:
        cfg.seed = args.seed
        cfg.seed_in_path = True
        cfg.__post_init__()            # re-derive output_dir + hf_repo_id with the _s{seed} suffix
    persistence.install_signal_handlers()
    seed_everything(cfg.seed)

    examples = data.load_sft_examples(cfg)
    if args.max_examples:
        examples = examples[:args.max_examples]
    print(f"Loaded {len(examples)} seller-turn examples", flush=True)

    print(f"Loading base model: {cfg.model_name}", flush=True)
    model, tokenizer = model_mod.load_base(cfg)

    if args.inspect:
        inspect_example(examples, tokenizer, cfg.max_seq_length)
        return

    model = model_mod.add_lora(model, cfg)
    dataset = build_dataset(examples, tokenizer, cfg.max_seq_length)
    print(f"Tokenised {len(dataset)} examples", flush=True)
    persistence.write_run_config(cfg)

    targs = TrainingArguments(
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        warmup_ratio=cfg.warmup_ratio,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        max_grad_norm=1.0,            # pinned so an HF default change can't drift it

        lr_scheduler_type="cosine",
        optim="adamw_torch",          # 32-bit AdamW
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=cfg.keep_last_k_checkpoints or None,   # bound epoch checkpoints
        seed=cfg.seed,
        output_dir=cfg.output_dir,
        report_to="none",
        bf16=(cfg.dtype == "bfloat16"),
        fp16=(cfg.dtype != "bfloat16"),
    )
    class _StopFlagCallback(TrainerCallback):
        """Turn the SIGTERM/SIGINT stop flag into save-and-stop at the next step boundary."""
        def on_step_end(self, targs, state, control, **kwargs):
            if persistence.stop_requested():
                control.should_save = True
                control.should_training_stop = True
            return control

    trainer = Trainer(
        model=model, args=targs, train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, label_pad_token_id=-100, padding=True),
        callbacks=[_StopFlagCallback()],
    )
    ckpt = persistence.valid_hf_trainer_checkpoint(cfg.output_dir) if cfg.resume else None
    trainer.train(resume_from_checkpoint=ckpt)
    if persistence.stop_requested():
        # interrupted: checkpoint saved by the callback, but the unit is not complete
        print("  stop requested; checkpoint saved — exiting 130 for the orchestrator to resume.",
              flush=True)
        raise SystemExit(130)

    lora_dir = os.path.join(cfg.output_dir, "lora_final")
    model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)
    print(f"Saved LoRA adapter to {lora_dir}", flush=True)

    uploaded_ok = persistence.push_to_hub(cfg, cfg.epochs)
    persistence.maybe_stop_pod(cfg, uploaded_ok)
    if cfg.hf_repo_id and not uploaded_ok:
        # only this push mirrors lora_final; non-zero exit makes the orchestrator retry it
        print("  final HF push failed; exiting non-zero so the orchestrator retries the push.",
              flush=True)
        raise SystemExit(8)


if __name__ == "__main__":
    main()
