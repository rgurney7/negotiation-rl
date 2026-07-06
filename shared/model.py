"""Base model load, LoRA attach, and chat-template prompt render, shared by SFT/PPO/GRPO."""


def _assert_text_only(model):
    """Raise on a multimodal load. Must check the top-level config: get_text_config() always
    sees vision_config=None."""
    cfg = getattr(model, "config", None)
    if cfg is None:
        return
    if getattr(cfg, "vision_config", None) is not None:
        raise RuntimeError(
            f"{type(model).__name__} loaded a MULTIMODAL model (top-level vision_config populated); "
            "expected the text decoder only")


def fast_kernels_available():
    """True if causal-conv1d + flash-linear-attention are importable (find_spec; no import)."""
    import importlib.util
    try:
        return all(importlib.util.find_spec(m) is not None for m in ("causal_conv1d", "fla"))
    except Exception:
        return False   # broken/partial install -> unavailable


def check_fast_kernels(cfg, available=None):
    """Warn (or hard-fail with cfg.require_fast_kernels) when the linear-attention kernels are missing."""
    if available is None:
        available = fast_kernels_available()
    if available:
        print("load_base: fast linear-attention kernels present (causal-conv1d + fla).", flush=True)
        return True
    msg = ("fast linear-attention kernels (causal-conv1d + flash-linear-attention) are NOT importable "
           "-> Qwen3.5 GatedDeltaNet layers run the SLOW torch fallback. "
           "Install: pip install causal-conv1d flash-linear-attention --no-build-isolation")
    if getattr(cfg, "require_fast_kernels", False):
        raise RuntimeError("load_base: " + msg)
    print("  WARN: " + msg, flush=True)
    return False


def load_base(cfg):
    """Load the text decoder + tokenizer for cfg.model_name. Returns (model, tokenizer)."""
    try:
        # Tier A: unsloth text-only loader (keeps unsloth's fast LoRA kernels).
        from unsloth import FastLanguageModel
        model, tok = FastLanguageModel.from_pretrained(
            model_name=cfg.model_name,
            max_seq_length=cfg.max_seq_length,
            dtype=cfg.torch_dtype,
            load_in_4bit=cfg.load_in_4bit,
            full_finetuning=False,
            trust_remote_code=cfg.trust_remote_code,
        )
        tokenizer = getattr(tok, "tokenizer", tok)        # FastLanguageModel may hand back a processor
        _assert_text_only(model)
        model.config._neg_loader = "unsloth"
        print("load_base: unsloth FastLanguageModel (text tower)", flush=True)
    except Exception as e:
        # OOM/auth/network failures aren't fixed by Tier B; re-raise
        if e.__class__.__name__ in (
                "OutOfMemoryError", "HfHubHTTPError", "HTTPError",
                "GatedRepoError", "RepositoryNotFoundError", "LocalEntryNotFoundError"):
            raise
        # Tier B: plain transformers. AutoModelForCausalLM defaults to CPU and nothing downstream
        # moves it, so pin to GPU when one is available.
        print(f"load_base: FastLanguageModel path unusable ({type(e).__name__}: {e}); "
              "falling back to transformers AutoModelForCausalLM", flush=True)
        import torch as _torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        _load_kw = {"dtype": cfg.torch_dtype, "trust_remote_code": cfg.trust_remote_code}
        if _torch.cuda.is_available():
            _load_kw["device_map"] = {"": "cuda"}
        model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **_load_kw)
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name, trust_remote_code=cfg.trust_remote_code)
        _assert_text_only(model)
        model.config._neg_loader = "hf"
        print("load_base: transformers AutoModelForCausalLM (text tower)", flush=True)

    # The default flex_attention path crashes at generation under torch 2.9.1 / transformers 5.5.0;
    # force sdpa.
    try:
        model.config._attn_implementation = "sdpa"
    except Exception:
        pass
    check_fast_kernels(cfg)
    gc = getattr(model, "generation_config", None)
    if gc is not None:
        print(f"load_base: checkpoint generation_config sampling defaults: "
              f"top_k={getattr(gc, 'top_k', None)} top_p={getattr(gc, 'top_p', None)} "
              f"temperature={getattr(gc, 'temperature', None)} min_p={getattr(gc, 'min_p', None)} "
              f"repetition_penalty={getattr(gc, 'repetition_penalty', None)} "
              "(neutralized at RL sampling call sites)", flush=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def add_lora(model, cfg):
    """Attach the LoRA adapter, using the peft path matching how the base was loaded."""
    if getattr(model.config, "_neg_loader", "unsloth") == "unsloth":
        from unsloth import FastLanguageModel
        return FastLanguageModel.get_peft_model(
            model,
            r=cfg.lora_r,
            target_modules=list(cfg.target_modules),
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=cfg.seed,
        )
    # Tier-B: plain peft LoRA + gradient checkpointing.
    from peft import LoraConfig, get_peft_model
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    return get_peft_model(model, LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, target_modules=list(cfg.target_modules),
        lora_dropout=cfg.lora_dropout, bias="none", task_type="CAUSAL_LM"))


def for_inference(model):
    """Switch to generation mode: unsloth fast-inference vs plain .eval()."""
    if getattr(model.config, "_neg_loader", "unsloth") == "unsloth":
        from unsloth import FastModel
        FastModel.for_inference(model)
    else:
        model.eval()


def for_training(model):
    """Switch to training mode: unsloth vs plain .train()."""
    if getattr(model.config, "_neg_loader", "unsloth") == "unsloth":
        from unsloth import FastModel
        FastModel.for_training(model)
    else:
        model.train()


def build_prompt(tokenizer, system_prompt, obs_text):
    """system + observation -> prompt via the chat template, non-reasoning where the template allows."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": obs_text},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
