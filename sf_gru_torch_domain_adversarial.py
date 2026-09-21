"""
sf_gru_torch_domain_adversarial.py
=====================================
Proposed fix for the cross-dataset generalization failure diagnosed in
cross_dataset_modality_ablation.py / cross_dataset_fusion_eval.py: the raw
bounding-box trajectory feature ("box") is the dominant driver of
PIE<->JAAD transfer failure, plausibly because it encodes dataset-specific
statistics (camera mounting height, frame rate, resolution, annotation
convention) rather than transferable motion signal. Simply excluding box
from training (train_full_{pie,jaad}_noboxspeed.py) is a partial,
asymmetric mitigation that costs real in-domain accuracy on PIE
(Table I of the paper).

This module instead trains box's representation to be domain-invariant
via a Domain-Adversarial Neural Network (DANN; Ganin & Lempitsky 2015):
a small domain classifier tries to predict which dataset (PIE vs. JAAD)
a sample came from, using ONLY the final fused hidden state (which the
modality ablation showed is dominated by box); a Gradient Reversal Layer
(GRL) between the fusion output and the domain classifier means the
main network is trained to make that prediction as HARD as possible,
pushing the box-dominated representation toward dataset-invariance while
the primary crossing-intent objective is trained normally (not
reversed). This is a genuine architectural intervention, not just an
ablation: unlike box exclusion, it keeps box's motion information and
tries to strip only the dataset-identifying component of it.

Requires joint PIE+JAAD training data (both datasets' train splits
combined into one batch stream, each labeled with a domain id) --- see
train_domain_adversarial.py.

Architecture
------------
  Same per-modality GRU stack as base SF-GRU (StackedGRU in
  sf_gru_torch.py), reusing the same modality order and fusion topology.
  The only addition: a domain classifier head reading the final GRU's
  hidden state through a Gradient Reversal Layer, trained jointly via
  domain cross-entropy with a lambda that anneals over training
  (standard DANN schedule, Ganin & Lempitsky 2015, Eq. 4-6).
"""

import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReversalFunction(Function):
    """Identity in the forward pass; negates (and scales by lambda) the
    gradient in the backward pass. This is the entire mechanism that
    makes 'domain adversarial' training possible without a separate
    minimax optimization loop -- the main network's optimizer step
    already includes the (negated) domain-classification gradient, so a
    single backward() call trains the feature extractor to CONFUSE the
    domain classifier while the domain classifier's own parameters are
    trained normally to succeed against a fixed feature extractor at
    that instant (Ganin & Lempitsky 2015, Sec 4)."""

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def set_lambda(self, lambda_):
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class DomainAdversarialStackedGRU(nn.Module):
    """Same topology as sf_gru_torch.StackedGRU (one GRU per modality,
    stacked-fusion), plus a domain classifier head on the final fused
    hidden state, reached through a Gradient Reversal Layer."""

    def __init__(self, data_types, data_sizes, hidden_units, domain_hidden=64):
        super().__init__()
        self.data_types = data_types
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

    def forward(self, inputs, return_domain_logits=False):
        x = None
        for i, gru in enumerate(self.grus):
            is_last = (i == len(self.grus) - 1)
            if i == 0:
                seq_in = inputs[0]
            else:
                seq_in = torch.cat([x, inputs[i]], dim=2)
            out, h = gru(seq_in)
            x = out if not is_last else h.squeeze(0)

        crossing_logit = self.output(x)
        crossing_prob = torch.sigmoid(crossing_logit)

        if return_domain_logits:
            reversed_features = self.grl(x)
            domain_logit = self.domain_classifier(reversed_features)
            return crossing_prob, domain_logit
        return crossing_prob
