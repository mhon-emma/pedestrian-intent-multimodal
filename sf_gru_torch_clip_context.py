"""
sf_gru_torch_clip_context.py
================================
Third backbone-specificity check (after VGG16 vs. DINOv2,
sf_gru_torch_dinov2_context.py, which found local_context's leakage
does NOT drop under DINOv2 -- separability stays at or above VGG16's).
Adds CLIP's vision encoder as a third, independently-motivated
backbone: a different training objective again (contrastive image-text
alignment, as opposed to VGG16's supervised ImageNet classification or
DINOv2's self-distillation) and a different architecture (ViT, like
DINOv2, but trained for a semantically different task entirely -- CLIP
features are explicitly shaped to align with natural-language
descriptions, not to reconstruct or classify the image itself).

If local_context separability stays high under CLIP too, that is a
third independent data point for "this is a property of the
surrounding-scene crop's content, not any one encoder's statistics" --
strengthening (though with diminishing marginal value after DINOv2
already agreed) the paper's existing backbone-specificity finding.

Same subclassing pattern as DinoV2ContextSFGRUTorch: overrides
_get_vgg/_vgg_forward only, reused across the SAME shape-compatibility
bridge (tile the pooled output across a trivial 2x2 spatial grid so the
base class's global-pooling code, which does squeeze + two successive
axis=0 averages, produces the correct unchanged vector) -- see
sf_gru_torch_dinov2_context.py's _vgg_forward docstring for why the
naive (1,1,1,D) reshape does NOT work and the tiling trick is needed.
"""

import numpy as np
import torch
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

from sf_gru_torch import SFGRUTorch

CLIP_MODEL_ID = 'openai/clip-vit-base-patch32'


class ClipContextSFGRUTorch(SFGRUTorch):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._clip = None
        self._clip_processor = None

    def _get_vgg(self):
        """Override: lazily load CLIP's vision tower instead of VGG16."""
        if self._clip is None:
            self._clip_processor = CLIPImageProcessor.from_pretrained(CLIP_MODEL_ID)
            model = CLIPVisionModelWithProjection.from_pretrained(CLIP_MODEL_ID)
            self._clip = model.to(self.device).eval()
            for p in self._clip.parameters():
                p.requires_grad = False
        return self._clip

    @torch.no_grad()
    def _vgg_forward(self, img_data):
        """Override: CLIP vision forward pass in place of VGG16's.
        Returns (1, 2, 2, 512) numpy array: CLIP's projected pooled
        output (image_embeds, 512-dim for ViT-B/32) tiled across a
        trivial 2x2 spatial grid -- see module docstring / this
        project's DINOv2 variant for why this shape bridge is needed
        for the base class's double-average global-pooling code to
        produce the correct, unchanged vector."""
        model = self._get_vgg()
        inputs = self._clip_processor(images=img_data, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = model(**inputs)
        pooled = outputs.image_embeds  # (1, 512) -- CLIP's projected pooled embedding
        vec = pooled.cpu().numpy().reshape(-1)  # (512,)
        tiled = np.tile(vec, (2, 2, 1))  # (2, 2, 512)
        feat = tiled[np.newaxis, ...]  # (1, 2, 2, 512)
        return feat
