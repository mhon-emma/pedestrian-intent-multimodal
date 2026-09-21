"""
sf_gru_torch_full_combo_data.py
==================================
Data-prep classes for the full-combo experiment (GRL + contrastive +
fine-tuned local_context). RawContextSFGRUTorch[JAAD] (raw-image
local_context) and ContrastiveScaleInvariantSFGRU (box-jittering for the
contrastive term) each override DIFFERENT SFGRUTorch methods --
load_images_crop_and_process vs. get_data_sequence[_balance]
respectively -- so they don't conflict and can be combined via ordinary
multiple inheritance; Python's MRO resolves each overridden method to
whichever parent actually defines it, with no method needing to call
the other parent's override.
"""

from sf_gru_torch_contrastive import ContrastiveScaleInvariantSFGRU
from sf_gru_torch_finetuned_context import RawContextSFGRUTorch, RawContextSFGRUTorchJAAD


class FullComboSFGRUTorch(RawContextSFGRUTorch, ContrastiveScaleInvariantSFGRU):
    """PIE-side (and generic PIE-path) data prep: raw local_context
    crops + box-jittering for the contrastive term, both active."""
    pass


class FullComboSFGRUTorchJAAD(RawContextSFGRUTorchJAAD, ContrastiveScaleInvariantSFGRU):
    """JAAD-side: JAAD's own path structure for local_context crops
    (RawContextSFGRUTorchJAAD's _context_save_folder override) +
    box-jittering for the contrastive term."""
    pass
