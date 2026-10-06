"""
sf_gru_torch_dinov2_context.py
==================================
Tests whether the cross-dataset leakage diagnosed in local_context
(Section~results-ablation: zeroing local_context HELPS transfer in most
architectures, the modality's removal is the single biggest
improvement any ablation produces) is a property of local_context AS A
CONCEPT (surrounding-scene visual appearance carries dataset-identity
information regardless of how it's encoded) or a property of THIS
PROJECT'S SPECIFIC ENCODER (a frozen, globally-pooled, ImageNet-trained
VGG16 conv feature map, which may encode low-level statistics --
JPEG/compression artifacts, sensor/lens characteristics, lighting
curves -- that are backbone-specific rather than inherent to the
local_context crop itself).

Method: swap the VGG16 feature extractor for a frozen DINOv2
(facebook/dinov2-base, ViT-B/14, self-distillation pretraining
explicitly designed to produce strong, transferable frozen features --
a meaningfully different training objective and architecture family
from VGG16's supervised ImageNet classification) for the local_context
crop ONLY. local_box, pose, and box are left completely unchanged --
isolating the test to exactly the modality under suspicion, rather than
re-testing the whole pipeline with a new backbone everywhere.

This subclasses SFGRUTorch and overrides ONLY the two methods that
touch the VGG16 extractor (_get_vgg, _vgg_forward); the rest of the
data pipeline (cropping, caching by (save_path, set_id, vid_id,
img_name) under a NEW cache namespace so this never collides with the
existing VGG16-backed cache, pose loading, box/speed sequences, the
whole training loop) is reused unchanged -- this is a swap of ONE
component, not a new pipeline.

DINOv2 outputs a 768-dim [CLS]-equivalent pooled embedding (base model,
patch14) rather than VGG16's (7,7,512) spatial grid -- _vgg_forward
reshapes this to (1,1,1,768) so the existing global-pooling code path
(_global_pooling='avg', averaging over a now-trivial 1x1 spatial grid)
still works unchanged; this is just a shape-compatibility reshape, not
a semantic pooling choice (DINOv2's own pooled output is already the
full image's global representation).
"""

import numpy as np
import torch
from transformers import AutoModel, AutoImageProcessor

from sf_gru_torch import SFGRUTorch

DINOV2_MODEL_ID = 'facebook/dinov2-base'


class DinoV2ContextSFGRUTorch(SFGRUTorch):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dinov2 = None
        self._dinov2_processor = None

    def _get_vgg(self):
        """Override: lazily load DINOv2 instead of VGG16. Named _get_vgg
        to match the base class's call site in _vgg_forward without
        needing to override load_images_crop_and_process itself."""
        if self._dinov2 is None:
            self._dinov2_processor = AutoImageProcessor.from_pretrained(DINOV2_MODEL_ID)
            model = AutoModel.from_pretrained(DINOV2_MODEL_ID)
            self._dinov2 = model.to(self.device).eval()
            for p in self._dinov2.parameters():
                p.requires_grad = False
        return self._dinov2

    @torch.no_grad()
    def _vgg_forward(self, img_data):
        """Override: DINOv2 forward pass in place of VGG16's. img_data:
        PIL Image, 224x224 (same crop/resize convention as the base
        class -- DINOv2's own processor will internally resize/normalize
        to its expected input size regardless, so the incoming 224x224
        is just this pipeline's existing convention, not a DINOv2
        requirement).

        Returns (1, 2, 2, 768) numpy array: DINOv2's pooled
        (CLS-equivalent) output tiled across a trivial 2x2 spatial grid
        so the base class's global-pooling code (_global_pooling='avg',
        which does np.squeeze then TWO successive axis=0 averages to
        collapse VGG16's native (7,7,512) spatial grid down to a
        512-dim vector) still produces the correct 768-dim vector
        unchanged -- averaging a tiled-constant grid returns the
        original value exactly, so this is a shape-compatibility
        bridge, not a new pooling operation. (A naive (1,1,1,768) reshape
        does NOT work here: np.squeeze collapses every size-1 axis,
        leaving a bare (768,) vector, and the pipeline's FIRST axis=0
        average then incorrectly averages across the 768 channels
        instead of a spatial axis, before the SECOND average crashes on
        the resulting 0-d scalar -- confirmed via a direct shape trace
        before this fix. Tiling to (2,2,768) keeps two real axes for
        squeeze to leave in place, matching VGG16's shape-after-squeeze
        structure (H, W, C) that the double-average logic assumes.)"""
        model = self._get_vgg()
        inputs = self._dinov2_processor(images=img_data, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = model(**inputs)
        pooled = outputs.pooler_output  # (1, 768)
        vec = pooled.cpu().numpy().reshape(-1)  # (768,)
        tiled = np.tile(vec, (2, 2, 1))  # (2, 2, 768) -- constant across the tiled grid
        feat = tiled[np.newaxis, ...]  # (1, 2, 2, 768), matches VGG16's (1, H, W, C) convention
        return feat
