"""PPO seller agent: LoRA policy plus a standalone value critic."""

from collections import namedtuple

import torch
import torch.nn as nn
from torch.distributions import Categorical

from shared import model as model_mod

Gen = namedtuple("Gen", ["prompt_ids", "gen_ids", "text"])


def build_critic(hidden_size):
    return nn.Sequential(
        nn.LayerNorm(hidden_size),
        nn.Linear(hidden_size, 4096), nn.GELU(),
        nn.Linear(4096, 2048), nn.GELU(),
        nn.Linear(2048, 1024), nn.GELU(),
        nn.Linear(1024, 1),
    )


def _assert_unit_sampling(cfg):
    """evaluate() scores the raw softmax; ratios are only unbiased at temperature=top_p=1.0."""
    assert cfg.temperature == 1.0 and cfg.top_p == 1.0, (
        f"log-prob eval assumes unit sampling (temp/top_p=1.0), got temp={cfg.temperature}, "
        f"top_p={cfg.top_p}; apply them in evaluate() before tuning.")


class PPOAgent:
    def __init__(self, cfg):
        self.cfg = cfg
        _assert_unit_sampling(cfg)
        base, self.tokenizer = model_mod.load_base(cfg)
        self.model = model_mod.add_lora(base, cfg)
        self.device = next(self.model.parameters()).device
        # multimodal configs keep hidden_size under text_config
        base_cfg = self.model.config
        text_cfg = base_cfg.get_text_config() if hasattr(base_cfg, "get_text_config") else base_cfg
        hidden_size = text_cfg.hidden_size
        # fp32 critic for stable value regression.
        self.critic = build_critic(hidden_size).to(self.device).float()

    def lora_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def critic_parameters(self):
        return list(self.critic.parameters())

    def build_prompt(self, seller_prompt, obs_text):
        return model_mod.build_prompt(self.tokenizer, seller_prompt, obs_text)

    def make_sampling_params(self, seed=None):
        return {"seed": self.cfg.seed if seed is None else seed}

    @torch.no_grad()
    def generate_batch(self, prompts, sampling_params):
        """One seller turn per prompt; returns unpadded prompt ids and gen ids trimmed at first eos."""
        seed = sampling_params.get("seed") if isinstance(sampling_params, dict) else None
        if seed is not None:
            torch.manual_seed(seed)
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        eos_id = self.tokenizer.eos_token_id

        prev_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        enc = self.tokenizer(prompts, return_tensors="pt", add_special_tokens=False,
                             padding=True).to(self.device)
        self.tokenizer.padding_side = prev_side

        model_mod.for_inference(self.model)
        # top_k=0/min_p=0/rep_penalty=1.0 neutralize generation_config defaults (Qwen ships top_k=20)
        # so sampling matches the full softmax evaluate() scores
        out = self.model.generate(
            **enc, max_new_tokens=self.cfg.max_new_tokens, do_sample=True,
            temperature=self.cfg.temperature, top_p=self.cfg.top_p,
            top_k=0, min_p=0.0, repetition_penalty=1.0, pad_token_id=pad_id)
        model_mod.for_training(self.model)

        plen = enc["input_ids"].shape[1]
        gens = []
        for i, _prompt in enumerate(prompts):
            keep = enc["attention_mask"][i].bool()
            prompt_ids = enc["input_ids"][i][keep].unsqueeze(0).clone()      # drop left padding
            gen_row = out[i, plen:]
            if eos_id is not None:                                           # trim past eos
                hit = (gen_row == eos_id).nonzero()
                if len(hit) > 0:
                    gen_row = gen_row[: hit[0, 0] + 1]
            gen_ids = gen_row.unsqueeze(0).clone()
            text = self.tokenizer.decode(gen_row, skip_special_tokens=True)
            gens.append(Gen(prompt_ids, gen_ids, text))
        return gens

    def evaluate(self, prompt_ids, gen_ids):
        """Token-level log-probs + entropy of the generated tokens, and V(s); critic input is detached."""
        ctx_len = prompt_ids.shape[1]
        n_gen = gen_ids.shape[1]
        full = torch.cat([prompt_ids, gen_ids], dim=1)

        out = self.model(full, output_hidden_states=True)
        logits = out.logits[:, ctx_len - 1: ctx_len - 1 + n_gen, :].float()
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(gen_ids)[0]
        entropy = dist.entropy()[0]

        last_ctx_hidden = out.hidden_states[-1][:, ctx_len - 1, :]            # pre-action state
        value = self.critic(last_ctx_hidden.detach().float()).squeeze()
        return log_probs, entropy, value

    def save_adapter(self):
        """Write a peft adapter dir (lora_final); save() writes the raw state dict for resume."""
        import os
        d = os.path.join(self.cfg.output_dir, "lora_final")
        self.model.save_pretrained(d)
        self.tokenizer.save_pretrained(d)
        print(f"  saved adapter: {d}", flush=True)

    def save(self, directory):
        import os
        from peft import get_peft_model_state_dict
        from safetensors.torch import save_file
        os.makedirs(directory, exist_ok=True)
        save_file(get_peft_model_state_dict(self.model), os.path.join(directory, "lora.safetensors"))
        torch.save(self.critic.state_dict(), os.path.join(directory, "critic.pt"))

    def load(self, directory):
        import os
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        set_peft_model_state_dict(self.model, load_file(os.path.join(directory, "lora.safetensors")))
        self.critic.load_state_dict(torch.load(os.path.join(directory, "critic.pt"), map_location=self.device))
