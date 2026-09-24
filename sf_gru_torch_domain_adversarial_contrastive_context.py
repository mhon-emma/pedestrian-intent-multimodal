"""
sf_gru_torch_domain_adversarial_contrastive_context.py
===========================================================
Completes the context-targeted domain-adversarial investigation
(sf_gru_torch_domain_adversarial_context.py) to the same standard as
this project's headline recipe: adds the box-embedding NT-Xent
contrastive tap used everywhere else in the GRL+contrastive family, and
is trained/evaluated at n=6 seeds both directions
(train_domain_adversarial_contrastive_context.py), rather than the n=3,
GRL-only run this model's plain predecessor received.

Motivation (unchanged from the plain predecessor): the corrected-label
per-modality zeroing ablation found that local_context, not box, is the
modality whose removal HELPS cross-dataset transfer in most
architectures -- the opposite of what the original (whole-
representation) GRL implicitly assumes by regularizing everything
equally. This module targets the GRL specifically at the local_context
branch's own hidden state (the same tap point as the plain predecessor)
while ALSO adding the box-embedding contrastive term (the same tap
point as every other GRL+contrastive variant in this project) -- a
different, more targeted domain-adversarial pressure than the
full-representation headline recipe, combined with the same
scale-invariance regularization on box that consistently helps
elsewhere. This tests whether targeting the GRL narrowly at the
diagnosed leakage source (rather than the whole fused representation)
does better, worse, or the same as the untargeted headline recipe.

Architecture: same per-modality GRU stack as StackedGRU. Two READ-ONLY
taps off different points of the stack, neither altering the main
classification forward path:
  - GRL + domain classifier on the local_context branch's own hidden
    state (same tap as ContextDomainAdversarialStackedGRU).
  - NT-Xent contrastive term on the box branch's own hidden state (same
    tap as DomainAdversarialContrastiveStackedGRU).
These are two independent branches' hidden states, so both taps can
coexist without interfering with each other or with the main fused
representation the classification head reads.
"""

import torch
import torch.nn as nn

from sf_gru_torch_domain_adversarial import GradientReversalLayer


class DomainAdversarialContrastiveContextStackedGRU(nn.Module):
    def __init__(self, data_types, data_sizes, hidden_units, domain_hidden=64):
        super().__init__()
        if 'local_context' not in data_types:
            raise ValueError(
                "requires 'local_context' in data_types -- got %r" % (data_types,))
        if 'box' not in data_types:
            raise ValueError("requires 'box' in data_types -- got %r" % (data_types,))
        self.data_types = data_types
        self._context_idx = data_types.index('local_context')
        self._box_idx = data_types.index('box')

        self.grus = nn.ModuleList()
        for i, size in enumerate(data_sizes):
            in_dim = size[-1] if i == 0 else hidden_units + size[-1]
            self.grus.append(nn.GRU(input_size=in_dim, hidden_size=hidden_units, batch_first=True))
        self.output = nn.Linear(hidden_units, 1)

        self.grl = GradientReversalLayer(lambda_=0.0)  # ramped up during training
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_units, domain_hidden),
            nn.ReLU(),
            nn.Linear(domain_hidden, 1),  # binary: source=0, target=1 (sigmoid applied in loss)
        )

    def forward(self, inputs, return_domain_logits=False, return_box_embedding=False):
        x = None
        context_hidden = None
        box_embed = None
        for i, gru in enumerate(self.grus):
            is_last = (i == len(self.grus) - 1)
            if i == 0:
                seq_in = inputs[0]
            else:
                seq_in = torch.cat([x, inputs[i]], dim=2)
            out, h = gru(seq_in)
            if i == self._context_idx:
                context_hidden = h.squeeze(0)
            if i == self._box_idx:
                box_embed = h.squeeze(0)
            x = out if not is_last else h.squeeze(0)

        crossing_logit = self.output(x)
        crossing_prob = torch.sigmoid(crossing_logit)

        outputs = [crossing_prob]
        if return_domain_logits:
            reversed_features = self.grl(context_hidden)
            domain_logit = self.domain_classifier(reversed_features)
            outputs.append(domain_logit)
        if return_box_embedding:
            outputs.append(box_embed)

        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)
