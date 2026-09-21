"""
cross_dataset_eval_homography.py
====================================
Cross-dataset generalization audit for the ground-plane-homography box
variant (train_full_pie_homography.py / train_full_jaad_homography.py)
-- see sf_gru_torch_homography.py for the fix being tested: box
position replaced by the pedestrian's foot-point trajectory projected
through a real ground-plane homography (PIE: built from PIE's own
published calibration; JAAD: built from an estimated default camera
height/pitch, since no official JAAD calibration exists).

IMPORTANT: uses HomographySFGRUTorch (not plain SFGRUTorch) for
evaluation too, AND must set method.DATASET correctly for the TARGET
dataset being evaluated on (not the source the model was trained on) --
test() internally calls get_data() -> get_data_sequence(), which reads
self.DATASET to select which camera projector (PIE-calibrated vs
JAAD-estimated) to apply to the box coordinates it's given. Evaluating
JAAD-trained models on PIE's test set must use PIE's real calibration
for the PROJECTION (since the pixel coordinates being projected are
PIE's own, regardless of which dataset trained the model weights).

Also re-applies BOTH get_pose and get_path per target dataset -- see
cross_dataset_eval_heightnorm.py's docstring for why this matters
(last-imported-module's get_path patch otherwise silently wins for
both directions, a real bug found and fixed earlier this session).

Requires: train_full_pie_homography.py and
train_full_jaad_homography.py have already been run.

Usage
-----
  python cross_dataset_eval_homography.py

Output
------
  results/cross_dataset_audit_homography.pkl
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
PIE_DATA_DIR = '/usr1/home/mehon/data_root/pie'
JAAD_UTIL_DIR = '/usr1/home/mehon/JAAD'
JAAD_DATA_DIR = '/usr1/home/mehon/JAAD'
RESULTS_DIR  = os.path.join(SFGRU_DIR, 'results')

sys.path.insert(0, SFGRU_DIR)
sys.path.append(PIE_UTIL_DIR)
sys.path.append(JAAD_UTIL_DIR)
os.chdir(SFGRU_DIR)

import train_full_pie_homography as _pie_mod
import train_full_jaad_homography_behonly as _jaad_mod

import sf_gru_torch as _sfgru_mod
import utils as _u
from pie_data import PIE
from jaad_data import JAAD
from sf_gru_torch_homography import HomographySFGRUTorch


def load_result_model_paths(results_pkl):
    with open(results_pkl, 'rb') as f:
        d = pickle.load(f)
    return [(r['model_path'], r.get('best_threshold', 0.5)) for r in d['runs']]


def apply_target_monkeypatches(target_name):
    if target_name == 'jaad':
        _sfgru_mod.SFGRUTorch.get_pose = _jaad_mod._safe_get_pose_jaad
        _sfgru_mod.SFGRUTorch.load_images_crop_and_process = _jaad_mod._load_images_crop_and_process_jaad
        _sfgru_mod.get_path = _jaad_mod._patched_get_path
        _u.get_path = _jaad_mod._patched_get_path
    else:
        _sfgru_mod.SFGRUTorch.get_pose = _pie_mod._safe_get_pose
        _sfgru_mod.get_path = _pie_mod._patched_get_path
        _u.get_path = _pie_mod._patched_get_path


def eval_cross(model_paths_thresholds, target_imdb, target_data_opts,
               source_name, target_name):
    apply_target_monkeypatches(target_name)
    beh_test = target_imdb.generate_data_trajectory_sequence('test', **target_data_opts)

    results = []
    for seed_idx, (model_path, threshold) in enumerate(model_paths_thresholds):
        method = HomographySFGRUTorch()
        # DATASET selects which camera projector applies to the box
        # pixel coordinates being processed -- that's determined by
        # the TARGET dataset (whose images/boxes we're evaluating on),
        # not the source the model weights were trained on.
        method.DATASET = target_name
        acc, auc, f1, prec, rec = method.test(beh_test, model_path, threshold=threshold)
        log.info('%s->%s seed=%d threshold=%.2f  Acc=%.4f  AUC=%.4f  F1=%.4f',
                 source_name, target_name, seed_idx, threshold, acc, auc, f1)
        results.append({'seed': seed_idx, 'acc': acc, 'auc': auc, 'f1': f1,
                        'prec': prec, 'rec': rec, 'model_path': model_path,
                        'threshold': threshold})
    return results


def summarise(runs):
    accs = np.array([r['acc'] for r in runs])
    aucs = np.array([r['auc'] for r in runs])
    f1s  = np.array([r['f1']  for r in runs])
    return {
        'acc_mean': accs.mean(), 'acc_std': accs.std(),
        'auc_mean': aucs.mean(), 'auc_std': aucs.std(),
        'f1_mean':  f1s.mean(),  'f1_std':  f1s.std(),
        'n': len(runs),
    }


def main():
    pie_model_paths = load_result_model_paths(
        os.path.join(RESULTS_DIR, 'full_pie_homography_results_rtmpose.pkl'))
    jaad_model_paths = load_result_model_paths(
        os.path.join(RESULTS_DIR, 'full_jaad_homography_behonly_results_rtmpose.pkl'))

    pie_imdb = PIE(data_path=PIE_DATA_DIR)
    jaad_imdb = JAAD(data_path=JAAD_DATA_DIR)

    log.info('=== PIE-trained (ground-plane homography box) models -> JAAD test set ===')
    pie_to_jaad = eval_cross(
        pie_model_paths, jaad_imdb, _jaad_mod.DATA_OPTS,
        source_name='pie', target_name='jaad')

    log.info('=== JAAD-trained (ground-plane homography box) models -> PIE test set ===')
    jaad_to_pie = eval_cross(
        jaad_model_paths, pie_imdb, _pie_mod.DATA_OPTS,
        source_name='jaad', target_name='pie')

    summary = {
        'pie_to_jaad': summarise(pie_to_jaad),
        'jaad_to_pie': summarise(jaad_to_pie),
    }

    out = os.path.join(RESULTS_DIR, 'cross_dataset_audit_homography_behonly.pkl')
    with open(out, 'wb') as f:
        pickle.dump({'pie_to_jaad_runs': pie_to_jaad, 'jaad_to_pie_runs': jaad_to_pie,
                    'summary': summary}, f)
    log.info('Saved: %s', out)

    print('\n' + '=' * 70)
    print(f'{"Direction":<16} {"Acc":>8} {"+/-":>6} {"AUC":>8} {"+/-":>6} {"F1":>8} {"+/-":>6}')
    print('-' * 70)
    for direction, s in summary.items():
        print(f'{direction:<16} {s["acc_mean"]:>8.4f} {s["acc_std"]:>6.4f} '
              f'{s["auc_mean"]:>8.4f} {s["auc_std"]:>6.4f} '
              f'{s["f1_mean"]:>8.4f} {s["f1_std"]:>6.4f}')
    print('=' * 70)


if __name__ == '__main__':
    main()
