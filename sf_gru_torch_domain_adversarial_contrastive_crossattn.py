"""
sf_gru_torch_domain_adversarial_contrastive_crossattn.py
============================================================
Second-architecture replication of the project's headline model
(DomainAdversarialContrastiveStackedGRU, sf_gru_torch_domain_adversarial_contrastive.py)
on CrossAttentionSFGRU's fusion topology (sf_gru_torch_fusion.py:
CrossAttentionStackedGRU) instead of the plain stacked-GRU fusion, to
test whether (a) the GRL+contrastive AUC improvement and (b) the
probe-measured separability/AUC dissociation both replicate on a
structurally distinct architecture, or are specific to SF-GRU's stacked
fusion.

Architecture: same per-modality GRU encoders + box-anchored
cross-attention + final GRU as CrossAttentionStackedGRU (each modality
independently encoded, 'box' as the attention query, all other
modalities concatenated as key/value context, attended output fed
through one more GRU, then a linear classifier head) -- see
sf_gru_torch_fusion.py's module docstring for the full original
rationale. Two taps are added on top, structurally identical in
purpose to DomainAdversarialContrastiveStackedGRU's taps, just reading
from this architecture's own intermediate tensors instead of the
stacked-GRU's:
  - GRL + domain classifier on the POST-ATTENTION, POST-FINAL-GRU fused
    hidden state (the same point CrossAttentionStackedGRU's forward
    pass hands to its own output head) -- the natural "final fused
    representation" analogue of DomainAdversarialContrastiveStackedGRU's
    x tap.
  - Contrastive NT-Xent tap on the box modality's OWN GRU encoder
    output (post-encoder, pre-attention -- the box encoder is one of
    self.encoders, structurally analogous to box's own GRU hidden state
    in the stacked variant), pooled over time (last timestep) to match
    the stacked variant's box_embed shape ([batch, hidden]).
"""

import torch
import torch.nn as nn

from sf_gru_torch_domain_adversarial import GradientReversalLayer
from sf_gru_torch_attention import _AttentionBlock


class CrossAttentionDomainAdversarialContrastiveStackedGRU(nn.Module):
    def __init__(self, data_types, data_sizes, hidden_units, domain_hidden=64, anchor='box'):
        super().__init__()
        if anchor not in data_types:
            raise ValueError(f'anchor modality {anchor!r} not in data_types {data_types}')
        self.data_types = data_types
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

        self.grl = GradientReversalLayer(lambda_=0.0)  # ramped up during training
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_units, domain_hidden),
            nn.ReLU(),
            nn.Linear(domain_hidden, 1),  # binary: source=0, target=1 (sigmoid applied in loss)
        )

    def forward(self, inputs, return_domain_logits=False, return_box_embedding=False):
        sequences = []
        box_embed = None
        for i, (gru, inp) in enumerate(zip(self.encoders, inputs)):
            out, _ = gru(inp)  # [batch, seq, hidden]
            sequences.append(out)
            if i == self.anchor_idx:
                box_embed = out[:, -1, :]  # last-timestep pooled, matches stacked variant's h.squeeze(0)

        anchor_seq = sequences[self.anchor_idx]
        others = [s for i, s in enumerate(sequences) if i != self.anchor_idx]
        context = torch.cat(others, dim=2)
        attended = self.cross_attn(anchor_seq, context)

        _, h = self.final_gru(attended)
        x = h.squeeze(0)

        crossing_logit = self.output(x)
        crossing_prob = torch.sigmoid(crossing_logit)

        outputs = [crossing_prob]
        if return_domain_logits:
            reversed_features = self.grl(x)
            domain_logit = self.domain_classifier(reversed_features)
            outputs.append(domain_logit)
        if return_box_embedding:
            outputs.append(box_embed)

        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)
