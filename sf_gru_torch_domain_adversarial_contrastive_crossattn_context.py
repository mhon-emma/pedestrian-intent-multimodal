"""
sf_gru_torch_domain_adversarial_contrastive_crossattn_context.py
=====================================================================
Extends the context-targeted domain-adversarial idea
(sf_gru_torch_domain_adversarial_contrastive_context.py: GRL reads only
the local_context branch's own hidden state, rather than the whole
fused representation -- found to match the full-representation
headline recipe on SF-GRU's stacked fusion, Section results-context) to
the cross-attention architecture
(sf_gru_torch_domain_adversarial_contrastive_crossattn.py).

Tests whether "narrow targeting matches broad regularization" is a
property of SF-GRU's stacked fusion specifically, or generalizes across
fusion mechanisms: does targeting the GRL at local_context's own
independent encoder output (before cross-attention, analogous to the
stacked variant's pre-concatenation tap) match cross-attention's
existing full-representation result (Table tab:crossattn:
PIE->JAAD 0.559+/-0.027, JAAD->PIE 0.446+/-0.063)?

Architecture: identical to CrossAttentionDomainAdversarialContrastiveStackedGRU
(box-anchored cross-attention fusion, box-embedding contrastive tap),
except the GRL + domain classifier now read local_context's own
independent encoder output (last timestep, pre-attention) instead of
the post-attention, post-final-GRU fused representation x.
"""

import torch
import torch.nn as nn

from sf_gru_torch_domain_adversarial import GradientReversalLayer
from sf_gru_torch_attention import _AttentionBlock


class CrossAttentionDomainAdversarialContrastiveContextStackedGRU(nn.Module):
    def __init__(self, data_types, data_sizes, hidden_units, domain_hidden=64, anchor='box'):
        super().__init__()
        if anchor not in data_types:
            raise ValueError(f'anchor modality {anchor!r} not in data_types {data_types}')
        if 'local_context' not in data_types:
            raise ValueError(f"requires 'local_context' in data_types -- got {data_types!r}")
        self.data_types = data_types
        self.anchor_idx = data_types.index(anchor)
        self.context_idx = data_types.index('local_context')

        self.encoders = nn.ModuleList([
            nn.GRU(input_size=size[-1], hidden_size=hidden_units, batch_first=True)
            for size in data_sizes
        ])
        num_modalities = len(data_sizes)
        context_dim = hidden_units * (num_modalities - 1)
        self.cross_attn = _AttentionBlock(hidden_units, context_dim=context_dim)
        self.final_gru = nn.GRU(input_size=hidden_units, hidden_size=hidden_units, batch_first=True)
        self.output = nn.Linear(hidden_units, 1)

        self.grl = GradientReversalLayer(lambda_=0.0)
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_units, domain_hidden),
            nn.ReLU(),
            nn.Linear(domain_hidden, 1),
        )

    def forward(self, inputs, return_domain_logits=False, return_box_embedding=False):
        sequences = []
        box_embed = None
        context_embed = None
        for i, (gru, inp) in enumerate(zip(self.encoders, inputs)):
            out, _ = gru(inp)
            sequences.append(out)
            if i == self.anchor_idx:
                box_embed = out[:, -1, :]
            if i == self.context_idx:
                context_embed = out[:, -1, :]  # local_context's OWN independent encoder output, pre-attention

        anchor_seq = sequences[self.anchor_idx]
        others = [s for i, s in enumerate(sequences) if i != self.anchor_idx]
        context_cat = torch.cat(others, dim=2)
        attended = self.cross_attn(anchor_seq, context_cat)

        _, h = self.final_gru(attended)
        x = h.squeeze(0)

        crossing_logit = self.output(x)
        crossing_prob = torch.sigmoid(crossing_logit)

        outputs = [crossing_prob]
        if return_domain_logits:
            reversed_features = self.grl(context_embed)  # GRL on local_context's branch, NOT fused x
            domain_logit = self.domain_classifier(reversed_features)
            outputs.append(domain_logit)
        if return_box_embedding:
            outputs.append(box_embed)

        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)
