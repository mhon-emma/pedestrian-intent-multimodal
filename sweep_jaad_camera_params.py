"""
sweep_jaad_camera_params.py
===============================
Sensitivity sweep on JAAD's ESTIMATED camera height/pitch, to isolate
whether the PIE->JAAD direction's continued near-chance performance
(with the ground-plane-homography fix, cross_dataset_audit_homography.pkl:
AUC 0.405, essentially unchanged from the 0.412 raw-pixel baseline) is
because our specific height=1.40m/pitch=-5deg estimate for JAAD is
simply wrong, or because the PIE->JAAD direction is stuck regardless of
what JAAD camera parameters we assume (which would point away from a
purely geometric/calibration explanation).

EVAL-ONLY: reuses the already-trained PIE homography model
(train_full_pie_homography.py's saved checkpoints) -- no retraining.
Only the projector used to interpret JAAD's box coordinates at TEST
time is swept, via HomographySFGRUTorch._jaad_projector_override
(see sf_gru_torch_homography.py).

Sweep grid: height in {1.2, 1.3, 1.4, 1.5, 1.6} m x pitch in
{-3, -5, -7, -10} deg = 20 combinations, covering the full plausible
range from the camera-mounting research (height 1.2-1.5m, pitch -3 to
-8deg) plus a couple of values slightly outside that range as a
robustness check (1.6m, -10deg matching PIE's own pitch).

Usage
-----
  python sweep_jaad_camera_params.py

Output
------
  results/jaad_camera_param_sweep.pkl
"""

import logging
import os
import pickle
import sys

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

SFGRU_DIR    = '/usr1/home/mehon/emma_pedestrian-intent-multimodal'
PIE_UTIL_DIR = '/usr1/home/mehon/PIE/utilities'
JAAD_UTIL_DIR = '/usr1/home/mehon/JAAD'
JAAD_DATA_DIR = '/usr1/home/mehon/JAAD'
RESULTS_DIR  = os.path.join(SFGRU_DIR, 'results')

sys.path.insert(0, SFGRU_DIR)
sys.path.append(PIE_UTIL_DIR)
sys.path.append(JAAD_UTIL_DIR)
os.chdir(SFGRU_DIR)

import train_full_pie_homography as _pie_mod
import train_full_jaad_homography as _jaad_mod

import sf_gru_torch as _sfgru_mod
import utils as _u
from jaad_data import JAAD
from sf_gru_torch_homography import (HomographySFGRUTorch, JAAD_K,
                                     _make_ground_projector, _make_undistorter)


def load_result_model_paths(results_pkl):
    with open(results_pkl, 'rb') as f:
        d = pickle.load(f)
    return [(r['model_path'], r.get('best_threshold', 0.5)) for r in d['runs']]


def apply_jaad_monkeypatches():
    _sfgru_mod.SFGRUTorch.get_pose = _jaad_mod._safe_get_pose_jaad
    _sfgru_mod.SFGRUTorch.load_images_crop_and_process = _jaad_mod._load_images_crop_and_process_jaad
    _sfgru_mod.get_path = _jaad_mod._patched_get_path
    _u.get_path = _jaad_mod._patched_get_path


def main():
    pie_model_paths = load_result_model_paths(
        os.path.join(RESULTS_DIR, 'full_pie_homography_results_rtmpose.pkl'))

    apply_jaad_monkeypatches()
    jaad_imdb = JAAD(data_path=JAAD_DATA_DIR)
    beh_test = jaad_imdb.generate_data_trajectory_sequence('test', **_jaad_mod.DATA_OPTS)

    heights = [1.2, 1.3, 1.4, 1.5, 1.6]
    pitches = [-3.0, -5.0, -7.0, -10.0]

    # JAAD has no distortion model (JAAD_D is all-zero -- see
    # sf_gru_torch_homography.py), so the undistorter is a no-op
    # regardless of height/pitch; only the ground projector needs to
    # vary per grid point.
    jaad_undistort = _make_undistorter(JAAD_K, np.array([0.0, 0.0, 0.0, 0.0]))

    results = {}
    for height in heights:
        for pitch in pitches:
            ground_fn = _make_ground_projector(height, pitch, JAAD_K)
            aucs = []
            for seed_idx, (model_path, threshold) in enumerate(pie_model_paths):
                method = HomographySFGRUTorch()
                method.DATASET = 'jaad'
                method._jaad_projector_override = (jaad_undistort, ground_fn)
                acc, auc, f1, prec, rec = method.test(beh_test, model_path, threshold=threshold)
                aucs.append(auc)
            mean_auc = np.mean(aucs)
            std_auc = np.std(aucs)
            results[(height, pitch)] = {'auc_mean': mean_auc, 'auc_std': std_auc, 'aucs': aucs}
            log.info('height=%.1fm pitch=%.0fdeg: AUC=%.4f+/-%.4f (seeds: %s)',
                     height, pitch, mean_auc, std_auc, [f'{a:.3f}' for a in aucs])

    out = os.path.join(RESULTS_DIR, 'jaad_camera_param_sweep.pkl')
    with open(out, 'wb') as f:
        pickle.dump(results, f)
    log.info('Saved: %s', out)

    print('\n' + '=' * 70)
    print(f'{"Height (m)":<12}{"Pitch (deg)":<14}{"AUC mean":<12}{"AUC std":<12}')
    print('-' * 70)
    best_key = max(results, key=lambda k: results[k]['auc_mean'])
    for (h, p), r in sorted(results.items(), key=lambda kv: -kv[1]['auc_mean']):
        marker = ' <-- best' if (h, p) == best_key else ''
        print(f'{h:<12.1f}{p:<14.0f}{r["auc_mean"]:<12.4f}{r["auc_std"]:<12.4f}{marker}')
    print('=' * 70)
    print(f'Best: height={best_key[0]}m pitch={best_key[1]}deg, AUC={results[best_key]["auc_mean"]:.4f}')
    print(f'Baseline (raw pixel, no fix): AUC=0.4123')
    print(f'Current default (height=1.40m, pitch=-5deg): AUC={results[(1.4, -5.0)]["auc_mean"]:.4f}')


if __name__ == '__main__':
    main()
