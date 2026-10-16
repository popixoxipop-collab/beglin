#!/usr/bin/env python3
"""P18 research-only shared-weight original BF16 gold-window engine candidate.

One original tensor stream, two independent unpadded causal sequences. Unlike
v1, explicit prompt lengths identify ALL teacher-forced gold positions (1..8)
including GSM8K; this avoids mistaking the penultimate token for an arbitrary
next-token inference. No changes to original decoder calculations.

DO NOT use for frozen P18 quality until original 27L/5291T development-only
real-model bit-exact parity has passed. No release/promotion functionality.
"""
from __future__ import annotations
import gc
import torch
from torch import nn
from torch.nn import functional as F


@torch.inference_mode()
def forward_two_original_gold_windows(teacher, token_sequences, prompt_lengths, *, max_length=1024):
    """Return two [gold_len, vocab_size] float32 tensors, matching P17 semantics.

    Each token sequence is prompt + teacher-forced gold; gold_len in 1..8.
    Requires a pristine original TensorStore. Caller is responsible for SHA
    pinning, memory/SSD/swap watchdogs, and never using held-out P18 test rows
    for performance tuning.
    """
    if (not isinstance(max_length, int) or isinstance(max_length, bool)
            or not 3 <= max_length <= 1024):
        raise ValueError("Unsupported P18 context maximum")
    if (not isinstance(token_sequences, (list, tuple)) or len(token_sequences) != 2
            or not isinstance(prompt_lengths, (list, tuple)) or len(prompt_lengths) != 2):
        raise ValueError("Exactly two sequences and two explicit prompt lengths required")
    if teacher.store.loaded_names:
        raise ValueError("Pristine TensorStore required for once-only source audit")
    ids, prompts, gold_counts = [], [], []
    for seq, prompt_length in zip(token_sequences, prompt_lengths):
        if not isinstance(seq, (list, tuple)) or not 3 <= len(seq) <= max_length:
            raise ValueError("Source sequence length outside original P18 limits")
        if (type(prompt_length) is not int or prompt_length < 2
                or not 1 <= len(seq) - prompt_length <= 8):
            raise ValueError("Invalid explicit teacher-forced gold window")
        if any(type(t) is not int or not 0 <= t < teacher.config.vocab_size for t in seq):
            raise ValueError("Invalid original tokenizer token ID")
        ids.append(torch.tensor([seq], dtype=torch.int64))
        prompts.append(prompt_length)
        gold_counts.append(len(seq) - prompt_length)
    mask, position = [], []
    for source in ids:
        n = source.shape[1]
        m = torch.full((1, 1, n, n), torch.finfo(teacher.dtype).min, dtype=teacher.dtype)
        mask.append(torch.triu(m, diagonal=1))
        position.append(torch.arange(n, dtype=torch.int64).unsqueeze(0))
    embedding = teacher.store.load("model.embed_tokens.weight", teacher.dtype)
    hidden = [F.embedding(source, embedding) for source in ids]
    del embedding
    gc.collect()
    for layer_index in range(teacher.config.num_hidden_layers):
        layer = teacher.layer(layer_index)
        for row_index in range(2):
            hidden[row_index] = layer(
                hidden[row_index], attention_mask=mask[row_index],
                position_ids=position[row_index], past_key_value=None,
                output_attentions=False, use_cache=False,
            )[0]
            if not bool(torch.isfinite(hidden[row_index]).all()):
                raise ValueError("Original decoder produced nonfinite activations")
        del layer
        gc.collect()
    norm = teacher.classes["DeepseekV2RMSNorm"](
        teacher.config.hidden_size, eps=teacher.config.rms_norm_eps,
    )
    norm.weight = nn.Parameter(
        teacher.store.load("model.norm.weight", teacher.dtype), requires_grad=False,
    )
    hidden = [norm(h) for h in hidden]
    del norm
    head = teacher.store.load("lm_head.weight", teacher.dtype)
    outputs = []
    for state, prompt_length, gold_len in zip(hidden, prompts, gold_counts):
        z = F.linear(state[:, prompt_length - 1:prompt_length + gold_len - 1, :],
                     head).to(torch.float32).squeeze(0).contiguous()
        if list(z.shape) != [gold_len, teacher.config.vocab_size] or not bool(torch.isfinite(z).all()):
            raise ValueError("Original full-vocabulary gold-window logits missing or nonfinite")
        outputs.append(z)
    del head, hidden
    loaded = teacher.store.loaded_names
    if len(loaded) != len(set(loaded)) or set(loaded) != set(teacher.store.mapping):
        raise ValueError("Original TensorStore must load all model tensors exactly once")
    teacher.store.verify_sources()
    return outputs
