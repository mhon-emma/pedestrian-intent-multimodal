"""
localbox_branch_probe_vgg16.py
==================================
Fills a gap needed to interpret dinov2_localbox_probe.py's result: this
project's existing probe (domain_leakage_probe.py) taps 'fused' and
'local_context' branch representations, but never 'local_box' branch
specifically with VGG16 -- so there was no VGG16 baseline number to
compare dinov2_localbox_probe.py's DINOv2-backed local_box branch
separability against. This script fills that one gap: reuses the
EXISTING VGG16-trained baseline checkpoint
(full_pie_nospeed_results_rtmpose.pkl's seed-0 model, the same
checkpoint domain_leakage_probe.py's 'baseline_fused'/
'baseline_local_context_branch' conditions use) -- no retraining, just
a different tap point on an already-trained model -- and extracts the
local_box branch's own hidden state instead.

Usage
-----
  python localbox_branch_probe_vgg16.py

Output
------
  results/localbox_branch_probe_vgg16.pkl
"""

import logging
import os
import pickle
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

SFGRU_DIR     = '/usr1/home/mehon/emma_pedestrian-intent-multimodal'
RESULTS_DIR   = os.path.join(SFGRU_DIR, 'results')

sys.path.insert(0, SFGRU_DIR)
sys.path.append('/usr1/home/mehon/PIE/utilities')
sys.path.append('/usr1/home/mehon/JAAD')
os.chdir(SFGRU_DIR)

from sf_gru_torch import StackedGRU
import domain_leakage_probe as _base


@torch.no_grad()
def extract_local_box_representation(model, inputs, data_types, device='cuda'):
    """Same forward-pass reimplementation pattern as
    domain_leakage_probe.py's extract_representation, tapping
    local_box's own GRU hidden state instead of local_context's."""
    model.eval()
    inputs_t = [torch.from_numpy(np.asarray(x)).float().to(device) for x in inputs]
    x = None
    box_hidden = None
    box_idx = data_types.index('local_box')
    for i, gru in enumerate(model.grus):
        is_last = (i == len(model.grus) - 1)
        seq_in = inputs_t[0] if i == 0 else torch.cat([x, inputs_t[i]], dim=2)
        out, h = gru(seq_in)
        if i == box_idx:
            box_hidden = h.squeeze(0)
        x = out if not is_last else h.squeeze(0)
    return box_hidden.cpu().numpy()


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    log.info('=== Extracting PIE test features ===')
    pie_inputs, pie_types, pie_sizes, _ = _base.get_test_features('pie')
    log.info('=== Extracting JAAD test features ===')
    jaad_inputs, jaad_types, jaad_sizes, _ = _base.get_test_features('jaad')
    assert pie_types == jaad_types, (pie_types, jaad_types)

    log.info('--- Baseline (PIE-trained, no adaptation): local_box branch hidden state ---')
    with open(os.path.join(RESULTS_DIR, 'full_pie_nospeed_results_rtmpose.pkl'), 'rb') as f:
        pie_baseline_paths = [r['model_path'] for r in pickle.load(f)['runs']]
    base_model, base_types = _base.load_checkpoint_model(pie_baseline_paths[0], StackedGRU)

    pie_repr_box = extract_local_box_representation(base_model, pie_inputs, pie_types, device=device)
    jaad_repr_box = extract_local_box_representation(base_model, jaad_inputs, jaad_types, device=device)

    result = _base.run_probe(pie_repr_box, jaad_repr_box, label='baseline_local_box_branch_vgg16')

    out = os.path.join(RESULTS_DIR, 'localbox_branch_probe_vgg16.pkl')
    with open(out, 'wb') as f:
        pickle.dump({'baseline_local_box_branch_vgg16': result}, f)
    log.info('Saved: %s', out)

    print('\n' + '=' * 80)
    print('VGG16 local_box branch separability (for comparison with DINOv2 local_box)')
    print('-' * 80)
    print(f"linear_acc={result['acc_mean']:.4f}+/-{result['acc_std']:.4f}  "
          f"mlp_acc={result['mlp_acc_mean']:.4f}+/-{result['mlp_acc_std']:.4f}")
    print('=' * 80)


if __name__ == '__main__':
    main()
