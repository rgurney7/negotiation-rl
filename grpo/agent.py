"""GRPO seller agent: LoRA policy over the frozen backbone, no critic."""

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from shared import model as model_mod


def _assert_unit_sampling(cfg):
    """evaluate scores the raw softmax; ratios are only unbiased at temperature=top_p=1.0."""
    assert cfg.temperature == 1.0 and cfg.top_p == 1.0, (
        f"log-prob eval assumes unit sampling (temp/top_p=1.0), got temp={cfg.temperature}, "
        f"top_p={cfg.top_p}; apply them in evaluate() before tuning.")


class GRPOAgent(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        _assert_unit_sampling(cfg)
        base, self.tokenizer = model_mod.load_base(cfg)
        self.llm = model_mod.add_lora(base, cfg)
        self.device = next(self.llm.parameters()).device
        self.tokenizer.padding_side = "right"

    def _prompt(self, system, obs_text):
        return model_mod.build_prompt(self.tokenizer, system, obs_text)

    def generate_group(self, system, obs_text, num_samples, seed=None):
        """Returns context_ids [1, C] and generated_ids [G, T] (right-padded)."""
        inputs = self.tokenizer(self._prompt(system, obs_text), return_tensors="pt",
                                add_special_tokens=False).to(self.device)
        if seed is not None:
            torch.manual_seed(seed)
        model_mod.for_inference(self.llm)
        with torch.no_grad():
            # top_k=0/min_p=0/rep_penalty=1.0 neutralize generation_config defaults (Qwen ships
            # top_k=20) so sampling matches the full softmax evaluate_batch scores
            output_ids = self.llm.generate(
                **inputs, max_new_tokens=self.cfg.max_new_tokens, do_sample=True,
                temperature=self.cfg.temperature, top_p=self.cfg.top_p,
                top_k=0, min_p=0.0, repetition_penalty=1.0,
                num_return_sequences=num_samples, pad_token_id=self.tokenizer.eos_token_id,
            )
        model_mod.for_training(self.llm)
        # clone off inference-mode tensors so autograd can track them later
        ctx = inputs.input_ids.clone()
        return ctx, output_ids[:, ctx.shape[1]:].clone()

    def decode(self, generated_ids):
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    def evaluate_batch(self, context_ids, generated_ids, mask):
        """Per-token log-probs + entropy for a group, [G, T]."""
        n, num_gen = generated_ids.shape
        input_len = context_ids.shape[1]
        context = context_ids.expand(n, input_len)
        full_ids = torch.cat([context, generated_ids], dim=1)
        attention_mask = torch.cat([torch.ones_like(context), mask.long()], dim=1)

        out = self.llm(full_ids, attention_mask=attention_mask)
        logits = out.logits[:, input_len - 1: input_len - 1 + num_gen, :]
        dist = Categorical(logits=logits.float())
        return dist.log_prob(generated_ids), dist.entropy()

    def evaluate_ref_batch(self, context_ids, generated_ids, mask):
        """Log-probs under the frozen base (adapter disabled): the reference policy."""
        self.llm.disable_adapter_layers()
        try:
            with torch.no_grad():
                ref_log_probs, _ = self.evaluate_batch(context_ids, generated_ids, mask)
        finally:
            self.llm.enable_adapter_layers()
        return ref_log_probs

    def lora_parameters(self):
        return [p for p in self.llm.parameters() if p.requires_grad]

    def save_adapter(self):
        """Write a peft adapter dir (lora_final); save() writes the raw state dict for resume."""
        import os
        d = os.path.join(self.cfg.output_dir, "lora_final")
        self.llm.save_pretrained(d)
        self.tokenizer.save_pretrained(d)
        print(f"  saved adapter: {d}", flush=True)

    def save(self, directory):
        import os
        from peft import get_peft_model_state_dict
        from safetensors.torch import save_file
        os.makedirs(directory, exist_ok=True)
        save_file(get_peft_model_state_dict(self.llm), os.path.join(directory, "lora.safetensors"))

    def load(self, directory):
        import os
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        set_peft_model_state_dict(self.llm, load_file(os.path.join(directory, "lora.safetensors")))
