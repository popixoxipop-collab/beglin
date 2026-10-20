"""Optional pinned Hugging Face BF16 adapter for G1 actual-model runs.
This module is importable without torch; importing it never downloads a model.
It must NEVER be used as evidence of real execution unless run receipts exist.
"""
from __future__ import annotations
import hashlib
import re
from dataclasses import dataclass

class AdapterContractError(RuntimeError):
    pass

PIN = re.compile(r"^[0-9a-f]{40}$")
PILOT = {
    "Qwen/Qwen2.5-1.5B-Instruct": (28, 14),
    "meta-llama/Llama-3.2-3B-Instruct": (28, 14),
    "microsoft/Phi-4-mini-instruct": (32, 16),
}

@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    revision: str
    num_layers: int
    layer_index: int

    def validate(self):
        if not PIN.fullmatch(self.revision):
            raise AdapterContractError("model revision must be a pinned 40-character git commit")
        if self.model_id not in PILOT:
            raise AdapterContractError("model not in approved 3-model pilot")
        if (self.num_layers, self.layer_index) != PILOT[self.model_id]:
            raise AdapterContractError("unexpected layer selection; paper appendix uses 4/8 depth")
        if not 0 <= self.layer_index < self.num_layers:
            raise AdapterContractError("layer index out of bounds")
        # NOTE: explicit zero-based layers[layer_index] convention for this adapter.
        # Inspect the author's implementation before claiming index-parity.
        return self

def blocks(model):
    """Supports only verified HF decoder stacks; no arbitrary module heuristics."""
    trunk = getattr(model, "model", None)
    result = getattr(trunk, "layers", None)
    if result is None:
        raise AdapterContractError("missing model.model.layers; model-specific mapping required")
    return result

def _first_hidden(output):
    hidden = output[0] if isinstance(output, tuple) else output
    if not hasattr(hidden, "ndim") or hidden.ndim != 3:
        raise AdapterContractError("expected rank-3 post-block residual activation")
    return hidden

class PinnedHF:
    def __init__(self, spec: ModelSpec, *, local_files_only: bool = True,
                 explicitly_allow_download: bool = False, device_map=None):
        self.spec = spec.validate()
        if not local_files_only and not explicitly_allow_download:
            raise AdapterContractError("network checkpoint download requires explicit opt-in")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise AdapterContractError("install torch and transformers in an isolated research venv") from exc
        self.torch = torch
        kwargs = dict(revision=spec.revision, local_files_only=local_files_only,
                      trust_remote_code=False)
        self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id, **kwargs)
        # Deliberately fail closed if Phi custom code is required; review that code separately.
        mk = dict(kwargs, use_safetensors=True, torch_dtype=torch.bfloat16)
        if device_map is not None:
            mk["device_map"] = device_map
        self.model = AutoModelForCausalLM.from_pretrained(spec.model_id, **mk)
        self.model.eval()
        if len(blocks(self.model)) != spec.num_layers:
            raise AdapterContractError("loaded model layer count differs from preregistered spec")
        self.block = blocks(self.model)[spec.layer_index]

    def _ids(self, text: str):
        if not isinstance(text, str) or not text:
            raise AdapterContractError("nonempty text required")
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _render(self, prompt: str):
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )

    def _input(self, prompt: str):
        ids = self._ids(self._render(prompt))
        if not ids:
            raise AdapterContractError("empty prompt encoding")
        return self.torch.tensor([ids], dtype=self.torch.long, device=self.model.device)

    def last_token(self, prompt: str):
        """Read-only final-token post-block residual, float32 NumPy."""
        holder = []
        def record(_module, _input, output):
            holder.append(_first_hidden(output)[:, -1, :].detach().float().cpu().numpy()[0].copy())
        hook = self.block.register_forward_hook(record)
        try:
            with self.torch.no_grad():
                ids = self._input(prompt)
                self.model(input_ids=ids, use_cache=False)
        finally:
            hook.remove()
        if len(holder) != 1:
            raise AdapterContractError(f"expected exactly one hook event; got {len(holder)}")
        return holder[0]

    def candidate_log_likelihood(self, prompt: str, answer: str, *,
                                 reduction: str = "sum", max_total_tokens: int = 1024):
        """Conditional log P(exact common answer string | model chat prompt).
        Tokenization at the prefix boundary MUST be stable; otherwise refuse.
        """
        if reduction not in {"sum", "mean"}:
            raise AdapterContractError("explicit reduction must be sum or mean")
        prefix = self._render(prompt)
        prefix_ids = self._ids(prefix)
        full_ids = self._ids(prefix + answer)
        n = len(prefix_ids)
        if n < 1 or full_ids[:n] != prefix_ids:
            raise AdapterContractError("tokenizer changed prefix encoding at answer boundary")
        if n == len(full_ids) or len(full_ids) > max_total_tokens:
            raise AdapterContractError("empty or too-long answer continuation")
        with self.torch.no_grad():
            ids = self.torch.tensor([full_ids], dtype=self.torch.long, device=self.model.device)
            logits = self.model(input_ids=ids, use_cache=False).logits[0].float()
            target = ids[0, n:]
            log_probs = logits[n-1:-1].log_softmax(dim=-1)
            result = log_probs.gather(1, target.unsqueeze(1)).sum()
        return float(result.cpu().item()) / (len(full_ids) - n if reduction == "mean" else 1)

    def generate(self, prompt: str, *, direction=None, alpha: float = 0.,
                 max_new_tokens: int = 32):
        """Inject post-block residual addition on every token, including decode."""
        if not 1 <= max_new_tokens <= 128:
            raise AdapterContractError("generation budget must be 1..128 tokens")
        torch = self.torch
        ids = self._input(prompt)
        hook = None
        if direction is not None and alpha != 0.:
            vec = torch.as_tensor(direction, dtype=torch.float32)
            if vec.ndim != 1 or len(vec) != self.model.config.hidden_size:
                raise AdapterContractError("direction hidden size mismatch")
            if not torch.isfinite(vec).all() or vec.norm() < 1e-10:
                raise AdapterContractError("nonfinite/zero direction")
            vec = vec / vec.norm()
            def steer(_module, _inputs, output):
                h = _first_hidden(output)
                delta = (float(alpha) * vec).to(device=h.device, dtype=h.dtype)
                updated = h + delta.view(1, 1, -1)
                return (updated, *output[1:]) if isinstance(output, tuple) else updated
            hook = self.block.register_forward_hook(steer)
        try:
            with torch.no_grad():
                output = self.model.generate(
                    input_ids=ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
        finally:
            if hook is not None:
                hook.remove()
        return self.tokenizer.decode(output[0, ids.shape[1]:],
                                     skip_special_tokens=True)

def text_sha256(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()
