"""
sf_gru_torch_domain_adversarial_uncertainty_context.py
===========================================================
Extends the context-targeted domain-adversarial idea
(sf_gru_torch_domain_adversarial_contrastive_context.py, found to match
the full-representation headline recipe on SF-GRU's stacked fusion) to
uncertainty-weighted fusion
(sf_gru_torch_domain_adversarial_uncertainty.py:
DomainAdversarialContrastiveUncertaintyFusionStackedGRU) -- the
architecture where the full-representation recipe was found to
SIGNIFICANTLY HURT the strongest unadapted baseline of any architecture
tested (Table tab:uncertainty: PIE->JAAD 0.585->0.499, a significant
drop). This tests whether targeting the GRL narrowly at local_context's
own independent encoder output -- rather than the shared,
uncertainty-weighted fused representation every modality's prediction
confidence depends on -- avoids the harm the broad intervention causes
on this specific architecture.

Architecture: identical to DomainAdversarialContrastiveUncertaintyFusionStackedGRU
(independent per-modality GRU encoders, softmax(-log_var) inverse-
variance fusion, box-embedding contrastive tap), except the GRL +
domain classifier read local_context's own independent encoder output
directly, instead of the uncertainty-weighted fused representation.
Critically, this also means the GRL's adversarial pressure no longer
flows back through the log-variance heads that determine EVERY
modality's fusion weight (since fused = softmax(-log_var) @ encodings
depends on ALL modalities' log-variance heads, which the broad
intervention's gradient reversal necessarily touches via fused) --
only local_context's own encoder is adversarially regularized, leaving
the uncertainty-weighting mechanism itself (and every other modality's
encoder) outside the adversarial loop entirely. This is a strictly
narrower intervention than the full-representation variant in a sense
specific to this architecture, not just a different tap point.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from sf_gru_torch_domain_adversarial import GradientReversalLayer


class DomainAdversarialContrastiveUncertaintyContextStackedGRU(nn.Module):
    def __init__(self, data_types, data_sizes, hidden_units, domain_hidden=64):
        super().__init__()
        if 'box' not in data_types:
            raise ValueError("requires 'box' in data_types -- got %r" % (data_types,))
        if 'local_context' not in data_types:
            raise ValueError("requires 'local_context' in data_types -- got %r" % (data_types,))
        self.data_types = data_types
        self._box_idx = data_types.index('box')
        self._context_idx = data_types.index('local_context')

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
        context_embed = encodings[self._context_idx]

        log_var_stack = torch.cat(log_vars, dim=1)
        weights = F.softmax(-log_var_stack, dim=1)

        stacked = torch.stack(encodings, dim=1)
        fused = (weights.unsqueeze(-1) * stacked).sum(dim=1)

        crossing_logit = self.output(fused)
        crossing_prob = torch.sigmoid(crossing_logit)

        outputs = [crossing_prob]
        if return_domain_logits:
            reversed_features = self.grl(context_embed)  # GRL on local_context's own encoder, NOT fused
            domain_logit = self.domain_classifier(reversed_features)
            outputs.append(domain_logit)
        if return_box_embedding:
            outputs.append(box_embed)

        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)
