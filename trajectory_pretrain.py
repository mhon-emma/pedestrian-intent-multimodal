"""
trajectory_pretrain.py
==========================
Third investigation: does a box-trajectory representation PRETRAINED on
a larger, more diverse corpus than either dataset alone generalize
better cross-dataset than SF-GRU's from-scratch box GRU (trained on
only the source dataset's ~100-800 labeled sequences)? Motivated by
Gesnouin et al. 2022's finding that generic ImageNet/Sports1M-pretrained
VISUAL backbones generalize better under domain shift than
task-specific ones -- this tests whether the same effect holds for the
MOTION modality specifically, which no prior work (including this
project's other experiments) has tested.

We do not have access to a genuinely large external trajectory corpus
(no local Waymo Open Motion / Argoverse 2 download) -- see this
module's docstring note below for the honest scope of what "pretrained"
means here. This is a SELF-SUPERVISED pretrain on the POOLED box-delta
trajectories from BOTH PIE's and JAAD's full train splits (unlabeled --
crossing labels are never touched during pretraining), which is still a
larger and more diverse trajectory distribution than either dataset's
own labeled training pool alone gives the from-scratch box GRU in every
other experiment in this project. This is a weaker, locally-feasible
version of the original "pretrained on a large external motion corpus"
idea, not a claim of parity with a genuine Waymo/Argoverse pretrain.

Method: masked-frame reconstruction (BERT-style). Each box-delta
sequence (14 frames x 4 dims, PIE+JAAD's train splits pooled, source
dataset's crossing labels never used) has a random subset of frames
masked (replaced with a learned mask token); a small Transformer encoder
must reconstruct the masked frames' true (x,y,w,h) delta values from
the unmasked context. This is a standard, architecture-agnostic
self-supervised objective for learning a general-purpose motion
representation, analogous to how the frozen VGG16 backbone used
throughout this project for local_box/local_context was pretrained on
ImageNet rather than on PIE/JAAD directly.

After pretraining, the encoder's PER-TIMESTEP output (not just a
pooled/final embedding, so it can be spliced into SF-GRU's box GRU slot
at the same sequence length the rest of the stack expects) replaces the
raw 4-dim box-delta features SF-GRU's box GRU currently consumes. The
encoder is FROZEN when spliced into cross-dataset training
(pretrained_trajectory_sfgru.py) -- only the downstream fusion (the box
GRU itself, since it now takes the pretrained embedding as input rather
than raw deltas, plus everything after it) is fine-tuned cross-dataset,
matching the "generic pretrained backbone, task-specific fine-tuning"
pattern this experiment is testing the motion-modality analogue of.

Usage
-----
  python trajectory_pretrain.py

Output
------
  data/models/trajectory_pretrain/encoder.pt
"""

import logging
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

SFGRU_DIR     = '/usr1/home/mehon/emma_pedestrian-intent-multimodal'
PIE_UTIL_DIR  = '/usr1/home/mehon/PIE/utilities'
PIE_DATA_DIR  = '/usr1/home/mehon/data_root/pie'
JAAD_UTIL_DIR = '/usr1/home/mehon/JAAD'
JAAD_DATA_DIR = '/usr1/home/mehon/JAAD'

sys.path.insert(0, SFGRU_DIR)
sys.path.append(PIE_UTIL_DIR)
sys.path.append(JAAD_UTIL_DIR)
os.chdir(SFGRU_DIR)

import sf_gru_torch as _sfgru_mod
from pie_data import PIE
from jaad_data import JAAD
from sf_gru_torch import SFGRUTorch

import train_full_pie_nospeed as _pie_mod
import train_full_jaad_nospeed_behonly as _jaad_mod

MODEL_OPTS_BASE = {
    # Only 'box' requested -- get_data() only runs VGG16 feature
    # extraction for modalities actually present in obs_input_type, so
    # requesting local_box/local_context/pose here (as the downstream
    # cross-dataset training scripts do) would silently trigger full
    # per-frame VGG16 inference on both datasets' entire train+val
    # splits just to throw the result away, since this module only
    # ever reads the 'box' output. Confirmed this was happening (a
    # first attempt hung for 10+ minutes at the pose-lookup stage
    # before being killed) -- fixed by requesting only what is used.
    'obs_input_type': ['box'],
    'enlarge_ratio': 1.5,
    'pred_target_type': ['crossing'],
    'obs_length': 15,
    'time_to_event': 60,
    'normalize_boxes': True,
}

ENCODER_DIR = os.path.join(SFGRU_DIR, 'data', 'models', 'trajectory_pretrain')
os.makedirs(ENCODER_DIR, exist_ok=True)


def apply_dataset_monkeypatches(dataset):
    if dataset == 'pie':
        _sfgru_mod.SFGRUTorch.get_pose = _pie_mod._safe_get_pose
        _sfgru_mod.get_path = _pie_mod._patched_get_path
        _sfgru_mod.SFGRUTorch.load_images_crop_and_process = SFGRUTorch.load_images_crop_and_process
    elif dataset == 'jaad':
        _sfgru_mod.SFGRUTorch.get_pose = _jaad_mod._safe_get_pose_jaad
        _sfgru_mod.get_path = _jaad_mod._patched_get_path
        _sfgru_mod.SFGRUTorch.load_images_crop_and_process = _jaad_mod._load_images_crop_and_process_jaad
    else:
        raise ValueError(dataset)


def get_box_sequences(dataset, split, pose_backend='rtmpose'):
    """Returns the raw box-delta sequences ([N, 14, 4] array) for
    `dataset`'s `split`, via the SAME get_data() pipeline every other
    script in this project uses (so normalization/windowing exactly
    matches what SF-GRU's box GRU sees at train/eval time) -- crossing
    labels are extracted but never used here (pretraining is
    unsupervised)."""
    apply_dataset_monkeypatches(dataset)
    if dataset == 'pie':
        imdb = PIE(data_path=PIE_DATA_DIR)
        os.environ['PIE_POSE_BACKEND'] = pose_backend
        beh_data = imdb.generate_data_trajectory_sequence(split, **_pie_mod.DATA_OPTS)
    else:
        imdb = JAAD(data_path=JAAD_DATA_DIR)
        os.environ['JAAD_POSE_BACKEND'] = pose_backend
        beh_data = imdb.generate_data_trajectory_sequence(split, **_jaad_mod.DATA_OPTS)

    method = SFGRUTorch(device='cpu')  # box-only extraction, no need for GPU/VGG here
    model_opts = dict(MODEL_OPTS_BASE)
    model_opts['dataset'] = f'{dataset}_trajpretrain'
    key = 'train' if split in ('train', 'val') else 'test'
    data, data_types, data_sizes = method.get_data({key: beh_data}, model_opts)
    inputs, _labels = data[key]
    box_idx = data_types.index('box')
    return np.asarray(inputs[box_idx], dtype=np.float32)  # [N, 14, 4]


class TrajectoryTransformerEncoder(nn.Module):
    """Small Transformer encoder over box-delta sequences. Per-timestep
    output (not pooled) so it can be spliced into SF-GRU's box GRU slot
    at matching sequence length. A learned [MASK] token embedding
    replaces masked frames' input before the encoder; a linear
    reconstruction head (used only during pretraining, discarded after)
    predicts the true 4-dim delta for masked positions from the
    encoder's contextual output."""

    def __init__(self, input_dim=4, hidden_dim=64, n_heads=4, n_layers=2,
                seq_len=14, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.mask_token = nn.Parameter(torch.zeros(hidden_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 2,
            dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.recon_head = nn.Linear(hidden_dim, input_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x, mask=None):
        """x: [batch, seq, 4]. mask: [batch, seq] bool, True = masked
        (replaced with mask_token before encoding). Returns
        (per_timestep_embedding [batch, seq, hidden_dim],
         reconstruction [batch, seq, 4] or None if mask is None)."""
        h = self.input_proj(x)
        if mask is not None:
            h = torch.where(mask.unsqueeze(-1), self.mask_token.view(1, 1, -1), h)
        h = h + self.pos_embed
        h = self.encoder(h)
        recon = self.recon_head(h) if mask is not None else None
        return h, recon

    @torch.no_grad()
    def encode(self, x):
        """Inference-time use (splicing into SF-GRU): no masking, just
        the per-timestep contextual embedding."""
        self.eval()
        h, _ = self.forward(x, mask=None)
        return h


def pretrain(epochs=200, batch_size=64, lr=1e-3, mask_prob=0.2, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    log.info('Pooling box-delta trajectories: PIE train+val + JAAD train+val (unlabeled)...')
    pie_train = get_box_sequences('pie', 'train')
    pie_val = get_box_sequences('pie', 'val')
    jaad_train = get_box_sequences('jaad', 'train')
    jaad_val = get_box_sequences('jaad', 'val')
    pool = np.concatenate([pie_train, pie_val, jaad_train, jaad_val], axis=0)
    log.info('Pretraining pool: %d trajectories (PIE %d+%d, JAAD %d+%d) -- '
             'larger/more diverse than either dataset\'s own labeled train '
             'split alone.', len(pool), len(pie_train), len(pie_val),
             len(jaad_train), len(jaad_val))

    X = torch.from_numpy(pool).float()
    n = len(X)
    seq_len = X.shape[1]

    model = TrajectoryTransformerEncoder(seq_len=seq_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss(reduction='none')

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            batch = X[idx].to(device)
            bs = batch.shape[0]

            mask = torch.rand(bs, seq_len, device=device) < mask_prob
            # Guarantee at least one masked frame per sequence (otherwise
            # the reconstruction loss for that row is trivially empty).
            no_mask_rows = ~mask.any(dim=1)
            if no_mask_rows.any():
                forced_idx = torch.randint(0, seq_len, (no_mask_rows.sum(),), device=device)
                mask[no_mask_rows, forced_idx] = True

            optimizer.zero_grad()
            _, recon = model(batch, mask=mask)
            loss_per_elem = criterion(recon, batch).mean(dim=-1)  # [batch, seq]
            loss = (loss_per_elem * mask.float()).sum() / mask.float().sum().clamp(min=1)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 20 == 0 or epoch == 0:
            log.info('epoch=%d/%d  masked_recon_mse=%.6f', epoch + 1, epochs, epoch_loss / n_batches)

    out_path = os.path.join(ENCODER_DIR, 'encoder.pt')
    torch.save({'model_state_dict': model.state_dict(), 'seq_len': seq_len,
               'hidden_dim': model.hidden_dim}, out_path)
    log.info('Saved pretrained trajectory encoder: %s', out_path)
    return out_path


if __name__ == '__main__':
    pretrain()
