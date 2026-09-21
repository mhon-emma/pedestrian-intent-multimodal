"""
sf_gru_torch_full_combo.py
=============================
Combines all three independently-validated fix mechanisms into one
model, to test whether they stack:

  1. Domain-adversarial training (GRL on the fused representation) --
     train_domain_adversarial_behonly.py: PIE->JAAD 0.498->0.608,
     JAAD->PIE 0.402->0.587 (n=6). Probe-confirmed NOT to achieve
     genuine representational invariance -- works as a generic
     regularizer, not literal dataset-identity removal.
  2. Contrastive scale-invariance (NT-Xent on the box embedding vs. a
     synthetically scale-jittered view) -- when combined with #1
     (train_domain_adversarial_contrastive.py), PIE->JAAD improved
     further to 0.618+/-0.019 (n=6); JAAD->PIE ~matched standalone GRL
     at 0.583+/-0.073.
  3. Fine-tuned local_context (VGG16 block5 unfrozen + adapter, instead
     of a frozen never-adapted feature extractor) --
     train_full_{pie,jaad}_finetuned_context.py: PIE->JAAD alone (no
     GRL/contrastive) reached 0.593+/-0.025 -- the leakage probe's
     highest-priority target, validated independently. JAAD->PIE was
     unreliable (0.354+/-0.193) due to JAAD-side overfitting (9.5M
     trainable params against only 195 training samples) -- NOT
     evidence the mechanism doesn't work, see that experiment's
     documented caveat.

Architecture: DomainAdversarialContrastiveStackedGRU's GRU-stack
topology (GRL tap on the final fused hidden state + box-embedding tap
for the contrastive term) is the base; the local_context modality's GRU
input is now FineTunedContextEncoder's live VGG-block5+adapter output
(raw pixels in, trainable feature vector out) instead of a precomputed
frozen feature vector -- exactly the same substitution
FineTunedContextStackedGRU made to plain StackedGRU. All three taps
(domain classifier, box embedding, context encoder) coexist without
interference since they read/write different points of the same GRU
stack's forward pass.

Given JAAD-side overfitting was the documented failure mode for fix #3
alone, this module also exposes an optional `context_dropout` on the
adapter output (0.0 default = off) so the training script can add
regularization specifically for JAAD without changing PIE's config --
kept as a constructor arg rather than hardcoded, since whether it's
needed at all is an empirical question for this combined setting, not
assumed from the standalone fix #3 result.
"""

import torch
import torch.nn as nn

from sf_gru_torch_domain_adversarial import GradientReversalLayer
from sf_gru_torch_finetuned_context import FineTunedContextEncoder


class FullComboStackedGRU(nn.Module):
    def __init__(self, data_types, data_sizes, hidden_units, domain_hidden=64,
                context_adapter_dim=512, context_dropout=0.0):
        super().__init__()
        if 'box' not in data_types:
            raise ValueError("FullComboStackedGRU requires 'box' in data_types -- got %r" % (data_types,))
        if 'local_context' not in data_types:
            raise ValueError("FullComboStackedGRU requires 'local_context' in data_types -- got %r" % (data_types,))
        self.data_types = data_types
        self._box_idx = data_types.index('box')
        self._context_idx = data_types.index('local_context')

        self.context_encoder = FineTunedContextEncoder(adapter_dim=context_adapter_dim)
        self.context_dropout = nn.Dropout(context_dropout) if context_dropout > 0 else nn.Identity()

        self.grus = nn.ModuleList()
        for i, size in enumerate(data_sizes):
            if i == self._context_idx:
                in_dim = context_adapter_dim if i == 0 else hidden_units + context_adapter_dim
            else:
                in_dim = size[-1] if i == 0 else hidden_units + size[-1]
            self.grus.append(nn.GRU(input_size=in_dim, hidden_size=hidden_units, batch_first=True))
        self.output = nn.Linear(hidden_units, 1)

        self.grl = GradientReversalLayer(lambda_=0.0)  # ramped up during training
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_units, domain_hidden),
            nn.ReLU(),
            nn.Linear(domain_hidden, 1),  # binary: PIE=0, JAAD=1 (sigmoid applied in loss)
        )

    def forward(self, inputs, return_domain_logits=False, return_box_embedding=False):
        x = None
        box_embed = None
        for i, gru in enumerate(self.grus):
            is_last = (i == len(self.grus) - 1)
            raw_in = inputs[i]
            if i == self._context_idx:
                raw_in = self.context_dropout(self.context_encoder(raw_in))
            if i == 0:
                seq_in = raw_in
            else:
                seq_in = torch.cat([x, raw_in], dim=2)
            out, h = gru(seq_in)
            if i == self._box_idx:
                box_embed = h.squeeze(0)
            x = out if not is_last else h.squeeze(0)

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
