#!/usr/bin/env python3
"""Experimental P18 two-sequence original-model BF16 weight-sharing core.

Runs each source sequence with its OWN original-length causal mask, positions,
and activations.  Each original decoder-layer object is loaded only once and
applied sequentially to both rows.  There is NO right-padding or modified
attention mask and there is NO change to model equations.

RESEARCH/DEV ONLY: this function does not certify correctness.  A separate
real full-vocab parity study must prove byte-level equality to independently
run, SHA-pinned P17 original BF16 forwards before 480 frozen-test use.
"""
from __future__ import annotations
import gc
import torch
from torch import nn
from torch.nn import functional as F

@torch.inference_mode()
def forward_two_original_source_sequences(teacher, token_sequences, *, max_length=1024):
    """Returns [float32 original-vocab logits at last real token] for two rows.

    teacher: existing original P17 GoldWindowTeacher (or its P18 bounded copy)
    with original TensorStore/decoder layer and complete checkpoint.  The caller
    must enforce the original model and P17 source SHA, capacity and provenance.
    """
    if not isinstance(token_sequences, (list, tuple)) or len(token_sequences) != 2:
        raise ValueError("Exactly two original dev source sequences are required")
    store = teacher.store
    if store.loaded_names:
        raise ValueError("Require a pristine TensorStore to attest once-only loads")
    sequences = []
    for row in token_sequences:
        if not isinstance(row, (list, tuple)) or not 2 <= len(row) <= max_length:
            raise ValueError("Original sequence length outside limits")
        if any(type(token) is not int or not 0 <= token < teacher.config.vocab_size for token in row):
            raise ValueError("Token not an original vocabulary integer")
        sequences.append(torch.tensor([row], dtype=torch.int64))
    masks, positions = [], []
    for ids in sequences:
        n = ids.shape[1]
        positions.append(torch.arange(n, dtype=torch.int64).unsqueeze(0))
        mask = torch.full((1, 1, n, n), torch.finfo(teacher.dtype).min, dtype=teacher.dtype)
        masks.append(torch.triu(mask, diagonal=1))
    embedding = store.load("model.embed_tokens.weight", teacher.dtype)
    hidden = [F.embedding(row, embedding) for row in sequences]
    del embedding
    gc.collect()
    for index in range(teacher.config.num_hidden_layers):
        layer = teacher.layer(index)  # Load original layer tensors ONCE.
        for row_index in range(2):
            hidden[row_index] = layer(
                hidden[row_index], attention_mask=masks[row_index],
                position_ids=positions[row_index], past_key_value=None,
                output_attentions=False, use_cache=False,
            )[0]
            if not bool(torch.isfinite(hidden[row_index]).all()):
                raise ValueError("Nonfinite original hidden state")
        del layer
        gc.collect()
    norm = teacher.classes["DeepseekV2RMSNorm"](
        teacher.config.hidden_size, eps=teacher.config.rms_norm_eps,
    )
    norm.weight = nn.Parameter(store.load("model.norm.weight", teacher.dtype), requires_grad=False)
    hidden = [norm(state) for state in hidden]
    del norm
    head = store.load("lm_head.weight", teacher.dtype)
    outputs = [F.linear(state[:, -2:-1, :], head).to(torch.float32).squeeze(0).contiguous() for state in hidden]
    del head, hidden
    if any(list(out.shape) != [1, teacher.config.vocab_size] or not bool(torch.isfinite(out).all())
           for out in outputs):
        raise ValueError("Missing full original vocabulary logits")
    loaded = store.loaded_names
    if len(loaded) != len(set(loaded)) or set(loaded) != set(store.mapping):
        raise ValueError("Source tensors were not loaded exactly once")
    store.verify_sources()
    return outputs
