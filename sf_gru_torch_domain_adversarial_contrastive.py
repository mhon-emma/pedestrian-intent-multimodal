"""
sf_gru_torch_domain_adversarial_contrastive.py
==================================================
Investigation #1: do the two independently-working fix attempts stack?
Domain-adversarial training (train_domain_adversarial_behonly.py,
StackedGRU, full fused representation) gave PIE->JAAD 0.498->0.608,
JAAD->PIE 0.402->0.587. Contrastive scale-invariance training
(sf_gru_torch_contrastive.py, box branch only, single-dataset supervised
training, no domain adaptation) gave PIE->JAAD 0.498->0.558 on its own.
These target different mechanisms -- domain-adversarial regularizes
dataset-identity broadly across the fused representation via UDA (source
labeled + target unlabeled, adversarial); contrastive scale-invariance
makes the box embedding invariant to synthetic pixel-scale perturbations
via a same-dataset augmentation trick (no target data touched at all).
There's no structural reason they can't run simultaneously: this module
combines both mechanisms in ONE model class, and
train_domain_adversarial_contrastive.py's training loop runs UDA (source
crossing loss + source/target domain-adversarial loss) with the
contrastive NT-Xent term (source box-embedding vs. its scale-jittered
view) added into the same combined loss, all on the SAME batch of
labeled source data per step (target data used only for the domain
classifier, as in every other UDA script in this project).

Architecture: same StackedGRU topology as DomainAdversarialStackedGRU
(GRL + domain classifier on the final fused hidden state), PLUS
ContrastiveScaleGRU's box-embedding tap (the box modality's own GRU
hidden state, extracted before concatenation with the next modality) --
structurally just the union of the two existing model classes' forward
passes, since both taps read different points of the same GRU stack and
don't interfere with each other.
"""

import torch
import torch.nn as nn

from sf_gru_torch_domain_adversarial import GradientReversalLayer


class DomainAdversarialContrastiveStackedGRU(nn.Module):
    def __init__(self, data_types, data_sizes, hidden_units, domain_hidden=64):
        super().__init__()
        if 'box' not in data_types:
            raise ValueError(
                "DomainAdversarialContrastiveStackedGRU requires 'box' in "
                "data_types -- got %r" % (data_types,))
        self.data_types = data_types
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
            nn.Linear(domain_hidden, 1),  # binary: PIE=0, JAAD=1 (sigmoid applied in loss)
        )

    def forward(self, inputs, return_domain_logits=False, return_box_embedding=False):
        x = None
        box_embed = None
        for i, gru in enumerate(self.grus):
            is_last = (i == len(self.grus) - 1)
            if i == 0:
                seq_in = inputs[0]
            else:
                seq_in = torch.cat([x, inputs[i]], dim=2)
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
