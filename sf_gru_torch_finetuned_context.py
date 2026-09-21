"""
sf_gru_torch_finetuned_context.py
====================================
Targeted fix #2 (see domain_leakage_probe.py / the modality-ablation
findings this session): local_context features come from a FROZEN,
never-fine-tuned ImageNet-pretrained VGG16 (sf_gru_torch.py:_vgg_forward,
@torch.no_grad()). The leakage probe showed this exact branch carries
the strongest dataset-identity signal of any modality (baseline probe
accuracy 0.964 on the local_context branch alone, vs. 0.877 on the full
fused representation) -- plausibly because a frozen generic CNN
naturally encodes low-level domain-correlated statistics (lighting,
JPEG compression, background texture, camera/lens characteristics) that
have nothing to do with pedestrian crossing behavior, and a model that
can never adjust those features has no way to suppress that signal.

This module unfreezes ONLY VGG16's last conv block (block5, layers 24-30
in torchvision's vgg16().features Sequential -- ~7M params) plus adds a
small trainable adapter (Linear+ReLU) after global pooling. Blocks 1-4
(layers 0-23, ~7.6M params) stay frozen -- generic low-level
edge/texture features are unlikely to be the specific domain leak, and
keeping them frozen sharply reduces overfitting risk on JAAD's tiny
195-sample train set. No GRL/domain-adversarial pressure here --
isolating the fine-tuning effect first, per this session's plan (add
GRL only if fine-tuning alone measurably helps; investigation #4 already
showed architecture changes + GRL can interfere rather than compound).

Architecture change required: the base pipeline (sf_gru_torch.py
load_images_crop_and_process) runs VGG once during DATA PREP and caches
the resulting (7,7,512) feature map to disk -- frozen features can be
cached like this, but trainable ones can't (gradients need to flow
through VGG on every forward pass). This module instead caches the RAW
cropped 224x224x3 uint8 image (much cheaper to store and still avoids
redoing PIL crop/jitter/resize every epoch) and runs
block5+adapter+pooling live inside the model's forward() pass.

Usage: see train_full_{pie,jaad}_finetuned_context.py
"""

import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as tv_transforms
from PIL import Image, ImageDraw

from sf_gru_torch import SFGRUTorch, StackedGRU, _IMAGENET_MEAN, _IMAGENET_STD
from utils import img_pad, jitter_bbox, squarify, update_progress


class RawContextSFGRUTorch(SFGRUTorch):
    """Overrides load_images_crop_and_process: when called for
    local_context (identified by save_path containing 'local_context'),
    returns RAW 224x224x3 uint8 arrays instead of running VGG. All other
    modalities (local_box, pose, box) are untouched -- delegates to the
    base class's unmodified implementation for those, so local_box's own
    (separately frozen) VGG-feature cache is unaffected by this change.

    _context_save_folder is PIE-shaped (save_path/set_id/vid_id) by
    default -- JAAD's path structure has no set level
    (images/<vid>/<frame>.png), so RawContextSFGRUTorchJAAD below
    overrides just this one method rather than duplicating the whole
    crop/cache loop."""

    def _context_save_folder(self, save_path, imp):
        set_id = imp.split('/')[-3]
        vid_id = imp.split('/')[-2]
        return os.path.join(save_path, set_id, vid_id)

    def load_images_crop_and_process(self, img_sequences, bbox_sequences,
                                     ped_ids, save_path,
                                     data_type='train',
                                     crop_type='none',
                                     crop_mode='warp',
                                     crop_resize_ratio=2,
                                     regen_data=False):
        if 'local_context' not in save_path or crop_type != 'surround':
            return super().load_images_crop_and_process(
                img_sequences, bbox_sequences, ped_ids, save_path,
                data_type=data_type, crop_type=crop_type, crop_mode=crop_mode,
                crop_resize_ratio=crop_resize_ratio, regen_data=regen_data)

        print("Generating {} RAW local_context crops (for fine-tuning) crop_type={}\
              \nsave_path={}, ".format(data_type, crop_type, save_path))
        sequences = []
        bbox_seq = bbox_sequences.copy()
        i = -1
        for seq, pid in zip(img_sequences, ped_ids):
            i += 1
            update_progress(i / len(img_sequences))
            img_seq = []
            for imp, b, p in zip(seq, bbox_seq[i], pid):
                flip_image = False
                img_name = imp.split('/')[-1].split('.')[0]

                img_save_folder = self._context_save_folder(save_path, imp)
                img_save_path = os.path.join(img_save_folder, img_name + '_' + p[0] + '_raw.pkl')

                cached = None
                if os.path.exists(img_save_path) and not regen_data:
                    try:
                        with open(img_save_path, 'rb') as fid:
                            cached = pickle.load(fid)
                    except (pickle.UnpicklingError, EOFError):
                        cached = None
                if cached is not None:
                    raw_arr = cached
                else:
                    if 'flip' in imp:
                        imp = imp.replace('_flip', '')
                        flip_image = True
                    img_data = Image.open(imp).convert('RGB')
                    if flip_image:
                        img_data = img_data.transpose(Image.FLIP_LEFT_RIGHT)
                    # Exact same crop as the base class's crop_type='surround'
                    # path: gray out the pedestrian's own box, then crop the
                    # enlarged surrounding region -- so the raw image this
                    # module trains on matches what the frozen-VGG baseline
                    # was ever shown, byte-for-byte apart from being kept as
                    # pixels instead of immediately reduced to VGG features.
                    b_org = [b[0], b[1], b[2], b[3]]
                    bbox = jitter_bbox(imp, [b], 'enlarge', crop_resize_ratio)[0]
                    bbox = squarify(bbox, 1, img_data.size[0])
                    bbox = list(map(int, bbox[0:4]))
                    draw = ImageDraw.Draw(img_data)
                    draw.rectangle([b_org[0], b_org[1], b_org[2], b_org[3]], fill=(128, 128, 128))
                    del draw
                    cropped_image = img_data.crop(bbox)
                    img_data = img_pad(cropped_image, mode='pad_resize', size=224)

                    raw_arr = np.asarray(img_data, dtype=np.uint8)  # (224, 224, 3)
                    if not os.path.exists(img_save_folder):
                        os.makedirs(img_save_folder, exist_ok=True)
                    tmp_path = '%s.tmp.%d' % (img_save_path, os.getpid())
                    with open(tmp_path, 'wb') as fid:
                        pickle.dump(raw_arr, fid, pickle.HIGHEST_PROTOCOL)
                    os.replace(tmp_path, img_save_path)

                img_seq.append(raw_arr)
            sequences.append(img_seq)
        sequences = np.array(sequences)  # (n_samples, obs_length, 224, 224, 3) uint8
        return sequences


class FineTunedContextEncoder(nn.Module):
    """VGG16 block5 (layers 24-30, unfrozen) + global-avg-pool + a small
    trainable adapter, applied per-frame. blocks 1-4 (layers 0-23) stay
    frozen and are run under no_grad -- this halves the memory/compute
    of a full backward pass through VGG while still letting the
    task-relevant top-level features adapt."""

    def __init__(self, adapter_dim=512):
        super().__init__()
        vgg = tv_models.vgg16(weights='IMAGENET1K_V1')
        self.frozen_features = vgg.features[:24].eval()
        for p in self.frozen_features.parameters():
            p.requires_grad = False
        self.trainable_block5 = vgg.features[24:]  # requires_grad=True by default
        self.adapter = nn.Sequential(
            nn.Linear(512, adapter_dim), nn.ReLU(),
        )
        self.transform = tv_transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)
        self.output_dim = adapter_dim

    def forward(self, raw_imgs):
        """raw_imgs: (batch, obs_length, 224, 224, 3) uint8 tensor.
        Returns (batch, obs_length, adapter_dim)."""
        b, t, h, w, c = raw_imgs.shape
        x = raw_imgs.reshape(b * t, h, w, c).permute(0, 3, 1, 2).float() / 255.0
        x = self.transform(x)

        with torch.no_grad():
            x = self.frozen_features(x)
        x = self.trainable_block5(x)  # (b*t, 512, 7, 7), gradients flow

        x = x.mean(dim=(2, 3))  # global-avg-pool -> (b*t, 512), matches
                                # the base pipeline's _global_pooling='avg'
        x = self.adapter(x)  # (b*t, adapter_dim)
        return x.reshape(b, t, self.output_dim)


class FineTunedContextStackedGRU(nn.Module):
    """Same stacked-fusion topology as StackedGRU, except the
    local_context modality's raw pixel input is first run through
    FineTunedContextEncoder (trainable) before entering its GRU, instead
    of arriving as a precomputed frozen feature vector."""

    def __init__(self, data_types, data_sizes, hidden_units, adapter_dim=512):
        super().__init__()
        if 'local_context' not in data_types:
            raise ValueError(
                "FineTunedContextStackedGRU requires 'local_context' in "
                "data_types -- got %r" % (data_types,))
        self.data_types = data_types
        self._context_idx = data_types.index('local_context')

        self.context_encoder = FineTunedContextEncoder(adapter_dim=adapter_dim)

        self.grus = nn.ModuleList()
        for i, size in enumerate(data_sizes):
            if i == self._context_idx:
                in_dim = adapter_dim if i == 0 else hidden_units + adapter_dim
            else:
                in_dim = size[-1] if i == 0 else hidden_units + size[-1]
            self.grus.append(nn.GRU(input_size=in_dim, hidden_size=hidden_units, batch_first=True))
        self.output = nn.Linear(hidden_units, 1)

    def forward(self, inputs):
        x = None
        for i, gru in enumerate(self.grus):
            is_last = (i == len(self.grus) - 1)
            raw_in = inputs[i]
            if i == self._context_idx:
                raw_in = self.context_encoder(raw_in)
            if i == 0:
                seq_in = raw_in
            else:
                seq_in = torch.cat([x, raw_in], dim=2)
            out, h = gru(seq_in)
            x = out if not is_last else h.squeeze(0)
        return torch.sigmoid(self.output(x))


class FineTunedContextSFGRUTorch(RawContextSFGRUTorch):
    def build_model(self, data_types, data_sizes):
        return FineTunedContextStackedGRU(data_types, data_sizes,
                                          self._num_hidden_units).to(self.device)


class RawContextSFGRUTorchJAAD(RawContextSFGRUTorch):
    """JAAD path variant: images/<vid>/<frame>.png has no set level
    (unlike PIE's images/<set>/<vid>/<frame>.png), so only
    _context_save_folder needs overriding -- the crop/cache logic in
    load_images_crop_and_process (inherited unchanged) doesn't otherwise
    depend on path structure. For non-local_context modalities, this
    class's load_images_crop_and_process falls through to
    super().load_images_crop_and_process(), i.e. RawContextSFGRUTorch's
    -> SFGRUTorch's PIE-shaped implementation, which is WRONG for JAAD
    (three-level set_id/vid_id path split, not JAAD's two-level
    vid_id-only path) -- train_full_jaad_finetuned_context.py patches
    SFGRUTorch.load_images_crop_and_process at the class level (the
    same monkeypatch pattern used by every other JAAD training script
    in this project, e.g. train_full_jaad_nospeed_behonly.py's
    _load_images_crop_and_process_jaad) to the correct JAAD
    implementation BEFORE this class's local_context special-case
    fallback is ever reached, so the fallback correctly resolves to
    JAAD's own crop logic for local_box/pose rather than PIE's."""

    def _context_save_folder(self, save_path, imp):
        vid_id = imp.split('/')[-2]
        return os.path.join(save_path, vid_id)


class FineTunedContextSFGRUTorchJAAD(RawContextSFGRUTorchJAAD):
    def build_model(self, data_types, data_sizes):
        return FineTunedContextStackedGRU(data_types, data_sizes,
                                          self._num_hidden_units).to(self.device)
