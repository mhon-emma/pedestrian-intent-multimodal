"""
sf_gru_torch_pretrained_traj.py
===================================
Splices a FROZEN, self-supervised-pretrained trajectory encoder
(trajectory_pretrain.py's TrajectoryTransformerEncoder, pretrained on
pooled PIE+JAAD unlabeled box-delta trajectories) into SF-GRU's box
modality slot, in place of the raw 4-dim box-delta features the
from-scratch box GRU normally consumes. Combined with the project's
domain-adversarial + contrastive headline recipe
(sf_gru_torch_domain_adversarial_contrastive.py) so this experiment
tests the SAME UDA training protocol, only with the box modality's
INPUT representation replaced.

Architecture: identical stacked-fusion topology to
DomainAdversarialContrastiveStackedGRU (local_box -> local_context ->
pose -> box, GRL on the final fused state, NT-Xent contrastive tap on
the box GRU's own hidden state), EXCEPT the box GRU's input at each
timestep is the frozen encoder's per-timestep embedding
(hidden_dim-dimensional) instead of the raw 4-dim box delta. The
pretrained encoder's weights are frozen throughout (requires_grad=False,
.eval() held) -- only the box GRU (now consuming a different input
width) and everything downstream of it is trained, matching the
"generic pretrained backbone + task-specific fine-tuning" pattern used
for local_box/local_context's frozen VGG16 features elsewhere in this
project.
"""

import torch
import torch.nn as nn

from sf_gru_torch_domain_adversarial import GradientReversalLayer
from trajectory_pretrain import TrajectoryTransformerEncoder


class PretrainedTrajDomainAdversarialContrastiveStackedGRU(nn.Module):
    def __init__(self, data_types, data_sizes, hidden_units, encoder_path,
                domain_hidden=64):
        super().__init__()
        if 'box' not in data_types:
            raise ValueError("requires 'box' in data_types -- got %r" % (data_types,))
        self.data_types = data_types
        self._box_idx = data_types.index('box')

        checkpoint = torch.load(encoder_path, map_location='cpu')
        self.traj_encoder = TrajectoryTransformerEncoder(
            seq_len=checkpoint['seq_len'], hidden_dim=checkpoint['hidden_dim'])
        self.traj_encoder.load_state_dict(checkpoint['model_state_dict'])
        for p in self.traj_encoder.parameters():
            p.requires_grad = False
        self.traj_encoder.eval()
        encoder_out_dim = checkpoint['hidden_dim']

        self.grus = nn.ModuleList()
        for i, size in enumerate(data_sizes):
            if i == self._box_idx:
                raw_dim = encoder_out_dim  # box GRU now consumes the frozen encoder's embedding
            else:
                raw_dim = size[-1]
            in_dim = raw_dim if i == 0 else hidden_units + raw_dim
            self.grus.append(nn.GRU(input_size=in_dim, hidden_size=hidden_units, batch_first=True))
        self.output = nn.Linear(hidden_units, 1)

        self.grl = GradientReversalLayer(lambda_=0.0)
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_units, domain_hidden),
            nn.ReLU(),
            nn.Linear(domain_hidden, 1),
        )

    def train(self, mode=True):
        """Override to keep the frozen trajectory encoder permanently in
        eval mode (no dropout/batchnorm drift) regardless of the rest of
        the model's train/eval state."""
        super().train(mode)
        self.traj_encoder.eval()
        return self

    def forward(self, inputs, return_domain_logits=False, return_box_embedding=False):
        x = None
        box_embed = None
        for i, gru in enumerate(self.grus):
            is_last = (i == len(self.grus) - 1)
            if i == self._box_idx:
                with torch.no_grad():
                    raw_in = self.traj_encoder.encode(inputs[i])  # frozen: [batch, seq, hidden_dim]
            else:
                raw_in = inputs[i]

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
