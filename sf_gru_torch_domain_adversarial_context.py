"""
sf_gru_torch_domain_adversarial_context.py
=============================================
Second domain-adversarial attempt, retargeted based on the corrected-label
per-modality zeroing ablation (cross_dataset_eval_attention_behonly.py /
cross_dataset_fusion_eval_behonly.py, run after the JAAD sample_type='beh'
labeling-bug fix). That ablation found the OPPOSITE of what the original
sf_gru_torch_domain_adversarial.py assumed: zeroing 'box' costs 0.08-0.16
AUC in every one of 8 architectures (box carries genuine transferable
motion signal), while zeroing 'local_context' HELPS PIE->JAAD transfer in
6/8 architectures (mean +0.024 AUC) -- local_context is the leakage-
consistent modality, not box.

local_context is a frozen, never-fine-tuned ImageNet-pretrained VGG16
conv-feature map (sf_gru_torch.py:_vgg_forward), globally pooled and cached
to disk -- i.e. the leakage most plausibly enters through low-level visual
statistics (lighting, JPEG compression, background texture, camera/lens
characteristics) that a frozen generic backbone naturally encodes but that
have nothing to do with pedestrian crossing behavior. A GRL cannot reach
back into the frozen VGG weights (and re-running VGG feature extraction
per training step would be far more expensive than this codebase's
existing precompute-and-cache pipeline), so this module instead attaches
the GRL to the LOCAL_CONTEXT GRU BRANCH's own hidden state, right after
its own GRU layer and BEFORE concatenation with the next modality --
mirroring ContrastiveScaleGRU's approach in sf_gru_torch_contrastive.py of
pulling out one branch's own hidden state rather than the fully-fused
final representation. This forces the GRU that consumes the frozen VGG
features to learn a domain-invariant TRANSFORM of them, rather than
regularizing box/pose/local_box at all (they flow through unmodified).

This is a genuinely different experiment from the original whole-
representation GRL (sf_gru_torch_domain_adversarial.py), not a rerun of it:
different modality targeted, motivated by ablation evidence gathered AFTER
the original was built (and after that original result turned out to
predate the labeling-bug fix -- see train_domain_adversarial_behonly.py).

Architecture
------------
  Same per-modality GRU stack and stacked-fusion topology as StackedGRU.
  Modality order is fixed by MODEL_OPTS['obs_input_type']; this module
  assumes 'local_context' is present and locates its GRU by name via
  data_types (matching the pattern used to build the stack itself), not
  by a hardcoded index -- data_types order can vary by modality-ablation
  architecture variant, this module always trains a plain (non-ablated)
  4-modality stack, but index-by-name keeps it robust regardless.

  local_context GRU's own final hidden state -> GRL -> domain classifier
  (binary PIE=0 / JAAD=1, sigmoid applied in the loss). The GRU's raw
  output sequence (not its hidden state) continues to flow into the
  concatenation + next-modality GRU exactly as in the base class --
  the GRL branch is a read-only tap off the hidden state, it does not
  alter the forward path of the main classification head.

Requires joint PIE+JAAD training data (both datasets' train splits
combined into one batch stream, each labeled with a domain id) -- see
train_domain_adversarial_context.py (same protocol/UDA setup as
train_domain_adversarial_behonly.py, model swapped for this module's).
"""

import torch
import torch.nn as nn

from sf_gru_torch_domain_adversarial import GradientReversalLayer


class ContextDomainAdversarialStackedGRU(nn.Module):
    """Same topology as sf_gru_torch.StackedGRU, plus a domain classifier
    head reading ONLY the local_context branch's own hidden state (not
    the final fused representation), reached through a GRL."""

    def __init__(self, data_types, data_sizes, hidden_units, domain_hidden=64):
        super().__init__()
        if 'local_context' not in data_types:
            raise ValueError(
                "ContextDomainAdversarialStackedGRU requires 'local_context' "
                "in data_types -- got %r" % (data_types,))
        self.data_types = data_types
        self._context_idx = data_types.index('local_context')

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
        context_hidden = None
        for i, gru in enumerate(self.grus):
            is_last = (i == len(self.grus) - 1)
            if i == 0:
                seq_in = inputs[0]
            else:
                seq_in = torch.cat([x, inputs[i]], dim=2)
            out, h = gru(seq_in)
            if i == self._context_idx:
                # Tap the local_context branch's own hidden state here --
                # read-only for the domain head, does not alter the main
                # forward path (`out`, not `h`, still feeds the next GRU
                # exactly as in the unmodified base class, unless this IS
                # also the last GRU in the stack, matching the base
                # class's own is_last handling below).
                context_hidden = h.squeeze(0)
            x = out if not is_last else h.squeeze(0)

        crossing_logit = self.output(x)
        crossing_prob = torch.sigmoid(crossing_logit)

        if return_domain_logits:
            reversed_features = self.grl(context_hidden)
            domain_logit = self.domain_classifier(reversed_features)
            return crossing_prob, domain_logit
        return crossing_prob
