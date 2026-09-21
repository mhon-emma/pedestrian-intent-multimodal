"""
sf_gru_torch_domain_adversarial_uncertainty.py
==================================================
Investigation #4: does the domain-adversarial gain (train_domain_adversarial_behonly.py,
StackedGRU base architecture: PIE->JAAD 0.498->0.608, JAAD->PIE 0.402->0.587)
stack with the best-performing NON-adversarial architecture found in the
8-way architecture sweep (uncertainty-weighted fusion,
sf_gru_torch_fusion.py:UncertaintyWeightedFusionStackedGRU, PIE->JAAD 0.585
at the "None"-ablation row)? These are two independent levers -- training
OBJECTIVE (domain-adversarial regularization) vs. architecture (parallel
uncertainty-weighted fusion instead of sequential stacked-concatenation) --
so there's no a priori reason they should be redundant.

Architecture: identical to UncertaintyWeightedFusionStackedGRU (independent
per-modality GRU encoders, softmax(-log_var) inverse-variance fusion
weights, weighted sum -> sigmoid classifier), plus a domain classifier
head reading the FUSED representation (the same `fused` vector the
classifier head reads, post-uncertainty-weighting) through a Gradient
Reversal Layer -- the natural analog of DomainAdversarialStackedGRU's tap
point (final fused hidden state) for this architecture's different
topology.

Requires joint PIE+JAAD training data -- see
train_domain_adversarial_uncertainty.py (same UDA protocol as
train_domain_adversarial_behonly.py, model swapped for this module's).

Third-architecture replication note (added later): the original class
below (GRL only, no contrastive term) was run PIE->JAAD only, n=3
(0.550+/-0.026), and never combined with the project's headline
GRL+contrastive recipe or completed for JAAD->PIE. A third architecture
subclass, DomainAdversarialContrastiveUncertaintyFusionStackedGRU, adds
the same NT-Xent contrastive tap on the box modality's own encoder
output (this architecture's independent per-modality encoders already
give each modality, including box, its own final hidden state --
`encodings[box_idx]` before fusion -- so the tap point is even more
direct here than in the stacked-GRU or cross-attention variants, which
need an intermediate hidden state read mid-stack or mid-sequence).
Matches every other GRL+contrastive replication in this project:
default contrastive hyperparameters (cw=0.5, ct=0.2), n=6 seeds, both
directions, see train_domain_adversarial_contrastive_uncertainty.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from sf_gru_torch_domain_adversarial import GradientReversalLayer


class DomainAdversarialUncertaintyFusionStackedGRU(nn.Module):
    def __init__(self, data_types, data_sizes, hidden_units, domain_hidden=64):
        super().__init__()
        self.data_types = data_types
        self.encoders = nn.ModuleList([
            nn.GRU(input_size=size[-1], hidden_size=hidden_units, batch_first=True)
            for size in data_sizes
        ])
        self.log_var_heads = nn.ModuleList([
            nn.Linear(hidden_units, 1) for _ in data_sizes
        ])
        self.output = nn.Linear(hidden_units, 1)

        self.grl = GradientReversalLayer(lambda_=0.0)  # ramped up during training
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_units, domain_hidden),
            nn.ReLU(),
            nn.Linear(domain_hidden, 1),  # binary: PIE=0, JAAD=1 (sigmoid applied in loss)
        )

    def forward(self, inputs, return_domain_logits=False):
        encodings = []
        log_vars = []
        for gru, lv_head, inp in zip(self.encoders, self.log_var_heads, inputs):
            _, h = gru(inp)
            h = h.squeeze(0)
            encodings.append(h)
            log_vars.append(lv_head(h))

        log_var_stack = torch.cat(log_vars, dim=1)
        weights = F.softmax(-log_var_stack, dim=1)

        stacked = torch.stack(encodings, dim=1)
        fused = (weights.unsqueeze(-1) * stacked).sum(dim=1)

        crossing_logit = self.output(fused)
        crossing_prob = torch.sigmoid(crossing_logit)

        if return_domain_logits:
            reversed_features = self.grl(fused)
            domain_logit = self.domain_classifier(reversed_features)
            return crossing_prob, domain_logit
        return crossing_prob


class DomainAdversarialContrastiveUncertaintyFusionStackedGRU(nn.Module):
    """GRL + contrastive on uncertainty-weighted fusion -- see module
    docstring. Structurally identical to DomainAdversarialUncertaintyFusionStackedGRU
    except forward() can also return the box modality's own (pre-fusion)
    encoder output for the NT-Xent contrastive tap, matching
    DomainAdversarialContrastiveStackedGRU's box_embed convention."""

    def __init__(self, data_types, data_sizes, hidden_units, domain_hidden=64):
        super().__init__()
        if 'box' not in data_types:
            raise ValueError("requires 'box' in data_types -- got %r" % (data_types,))
        self.data_types = data_types
        self._box_idx = data_types.index('box')

        self.encoders = nn.ModuleList([
            nn.GRU(input_size=size[-1], hidden_size=hidden_units, batch_first=True)
            for size in data_sizes
        ])
        self.log_var_heads = nn.ModuleList([
            nn.Linear(hidden_units, 1) for _ in data_sizes
        ])
        self.output = nn.Linear(hidden_units, 1)

        self.grl = GradientReversalLayer(lambda_=0.0)
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_units, domain_hidden),
            nn.ReLU(),
            nn.Linear(domain_hidden, 1),
        )

    def forward(self, inputs, return_domain_logits=False, return_box_embedding=False):
        encodings = []
        log_vars = []
        for gru, lv_head, inp in zip(self.encoders, self.log_var_heads, inputs):
            _, h = gru(inp)
            h = h.squeeze(0)
            encodings.append(h)
            log_vars.append(lv_head(h))

        box_embed = encodings[self._box_idx]

        log_var_stack = torch.cat(log_vars, dim=1)
        weights = F.softmax(-log_var_stack, dim=1)

        stacked = torch.stack(encodings, dim=1)
        fused = (weights.unsqueeze(-1) * stacked).sum(dim=1)

        crossing_logit = self.output(fused)
        crossing_prob = torch.sigmoid(crossing_logit)

        outputs = [crossing_prob]
        if return_domain_logits:
            reversed_features = self.grl(fused)
            domain_logit = self.domain_classifier(reversed_features)
            outputs.append(domain_logit)
        if return_box_embedding:
            outputs.append(box_embed)

        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)
