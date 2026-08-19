"""
sf_gru_torch_fusion.py
========================
New multimodal fusion architectures, proposed to address a pattern observed
in the pose_backend x fusion_architecture sweep (sfgru_full_sweep_results.csv
in Pedestrian-intent): pose_attention/modality_fusion win on PIE while
cross_modal/other_modal win on JAAD, and none of the existing 4 architectures
(sf_gru_torch_attention.py) let the model learn to trust a modality
conditionally -- they all fuse unconditionally (concat, symmetric
self-attention). The hypothesis: JAAD is noisier/more occluded, so
architectures that can suppress an unreliable modality do better there,
while PIE's cleaner signal doesn't need it. These 3 variants each test a
different way of letting the model weight modalities adaptively:

  GatedFusionSFGRU
    Each modality encoded independently (GRU, return_sequences=False) into a
    vector. A per-modality scalar gate (sigmoid over a small MLP on the
    modality's own encoding) scales it before concatenation and
    classification -- a Gated Multimodal Unit (Arevalo et al. 2017) applied
    across all 5 modalities rather than just 2.

  CrossAttentionSFGRU
    Designates one modality as the anchor query (default: 'box', the most
    reliable trajectory signal) and lets every other modality serve as
    key/value in a cross-attention block, instead of the symmetric
    self-attention used by ModalityFusionSFGRU/CrossModalSFGRU. Closer to
    LXMERT/ViLBERT-style cross-modal transformers, where one stream anchors
    attention rather than all streams attending to each other equally.

  UncertaintyWeightedFusionSFGRU
    Each modality's encoder also predicts a scalar log-variance via an
    auxiliary head; modalities are fused via inverse-variance weighting
    (softmax over -log_var) before concatenation and classification. Directly
    tests whether the model can learn "trust this modality less when it's
    unreliable" from the data itself, without hand-picking an anchor modality
    or which modality to gate.

All three reuse SFGRUTorch's entire data pipeline unchanged (only
build_model() is overridden), same as sf_gru_torch_attention.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from sf_gru_torch import SFGRUTorch
from sf_gru_torch_attention import _AttentionBlock


# ---------------------------------------------------------------------------
# 1. Gated multimodal fusion
# ---------------------------------------------------------------------------

class GatedFusionStackedGRU(nn.Module):
    def __init__(self, data_types, data_sizes, hidden_units):
        super().__init__()
        self.encoders = nn.ModuleList([
            nn.GRU(input_size=size[-1], hidden_size=hidden_units, batch_first=True)
            for size in data_sizes
        ])
        self.gates = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_units, hidden_units), nn.Tanh(),
                          nn.Linear(hidden_units, 1))
            for _ in data_sizes
        ])
        num_modalities = len(data_sizes)
        self.output = nn.Linear(hidden_units * num_modalities, 1)

    def forward(self, inputs):
        gated = []
        for gru, gate, inp in zip(self.encoders, self.gates, inputs):
            _, h = gru(inp)
            h = h.squeeze(0)  # [batch, hidden]
            g = torch.sigmoid(gate(h))  # [batch, 1]
            gated.append(g * h)
        x = torch.cat(gated, dim=1)  # [batch, hidden*num_modalities]
        return torch.sigmoid(self.output(x))


class GatedFusionSFGRU(SFGRUTorch):
    def build_model(self, data_types, data_sizes):
        return GatedFusionStackedGRU(data_types, data_sizes, self._num_hidden_units).to(self.device)


# ---------------------------------------------------------------------------
# 2. Cross-attention with a designated anchor modality
# ---------------------------------------------------------------------------

class CrossAttentionStackedGRU(nn.Module):
    def __init__(self, data_types, data_sizes, hidden_units, anchor='box'):
        super().__init__()
        self.data_types = data_types
        if anchor not in data_types:
            raise ValueError(f'anchor modality {anchor!r} not in data_types {data_types}')
        self.anchor_idx = data_types.index(anchor)
        self.encoders = nn.ModuleList([
            nn.GRU(input_size=size[-1], hidden_size=hidden_units, batch_first=True)
            for size in data_sizes
        ])
        num_modalities = len(data_sizes)
        context_dim = hidden_units * (num_modalities - 1)
        self.cross_attn = _AttentionBlock(hidden_units, context_dim=context_dim)
        self.final_gru = nn.GRU(input_size=hidden_units, hidden_size=hidden_units, batch_first=True)
        self.output = nn.Linear(hidden_units, 1)

    def forward(self, inputs):
        sequences = []
        for gru, inp in zip(self.encoders, inputs):
            out, _ = gru(inp)  # [batch, seq, hidden]
            sequences.append(out)

        anchor_seq = sequences[self.anchor_idx]
        others = [s for i, s in enumerate(sequences) if i != self.anchor_idx]
        context = torch.cat(others, dim=2)  # [batch, seq, hidden*(num_modalities-1)]
        attended = self.cross_attn(anchor_seq, context)  # [batch, seq, hidden]

        _, h = self.final_gru(attended)
        x = h.squeeze(0)
        return torch.sigmoid(self.output(x))


class CrossAttentionSFGRU(SFGRUTorch):
    """anchor: which modality serves as the query (default 'box', the most
    reliable trajectory signal -- present and clean in both PIE and JAAD
    regardless of pose backend)."""

    def __init__(self, *args, anchor='box', **kwargs):
        super().__init__(*args, **kwargs)
        self.anchor = anchor

    def build_model(self, data_types, data_sizes):
        return CrossAttentionStackedGRU(data_types, data_sizes, self._num_hidden_units,
                                        anchor=self.anchor).to(self.device)


# ---------------------------------------------------------------------------
# 3. Uncertainty-weighted fusion
# ---------------------------------------------------------------------------

class UncertaintyWeightedFusionStackedGRU(nn.Module):
    def __init__(self, data_types, data_sizes, hidden_units):
        super().__init__()
        self.encoders = nn.ModuleList([
            nn.GRU(input_size=size[-1], hidden_size=hidden_units, batch_first=True)
            for size in data_sizes
        ])
        # auxiliary log-variance head per modality -- unconstrained scalar,
        # softmax(-log_var) below turns it into an inverse-variance weight
        self.log_var_heads = nn.ModuleList([
            nn.Linear(hidden_units, 1) for _ in data_sizes
        ])
        self.output = nn.Linear(hidden_units, 1)

    def forward(self, inputs):
        encodings = []
        log_vars = []
        for gru, lv_head, inp in zip(self.encoders, self.log_var_heads, inputs):
            _, h = gru(inp)
            h = h.squeeze(0)  # [batch, hidden]
            encodings.append(h)
            log_vars.append(lv_head(h))  # [batch, 1]

        log_var_stack = torch.cat(log_vars, dim=1)  # [batch, num_modalities]
        weights = F.softmax(-log_var_stack, dim=1)  # [batch, num_modalities], inverse-variance

        stacked = torch.stack(encodings, dim=1)  # [batch, num_modalities, hidden]
        fused = (weights.unsqueeze(-1) * stacked).sum(dim=1)  # [batch, hidden]
        return torch.sigmoid(self.output(fused))


class UncertaintyWeightedFusionSFGRU(SFGRUTorch):
    def build_model(self, data_types, data_sizes):
        return UncertaintyWeightedFusionStackedGRU(data_types, data_sizes,
                                                    self._num_hidden_units).to(self.device)
