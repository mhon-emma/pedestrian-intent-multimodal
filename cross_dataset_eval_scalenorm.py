"""
cross_dataset_eval_scalenorm.py
==================================
Cross-dataset generalization audit for the scale-normalized-box SF-GRU
variant (train_full_pie_scalenorm.py / train_full_jaad_scalenorm.py) --
see sf_gru_torch_scalenorm.py for the fix being tested: box displacement
normalized by the pedestrian's own apparent box size, instead of raw
pixel deltas, to remove camera/resolution-dependent scale as a
dataset-identifying signal.

IMPORTANT: uses ScaleNormSFGRUTorch (not plain SFGRUTorch) for
evaluation too -- test() internally calls get_data() ->
get_data_sequence(), so evaluating a scale-norm-trained checkpoint with
the base class would silently recompute box in raw pixels, corrupting
the comparison.

Requires: train_full_pie_scalenorm.py and train_full_jaad_scalenorm.py
have already been run.

Usage
-----
  python cross_dataset_eval_scalenorm.py

Output
------
  results/cross_dataset_audit_scalenorm.pkl
  Comparable directly against results/cross_dataset_audit.pkl (with-box,
  raw pixels), cross_dataset_audit_noboxspeed.pkl (box excluded), and
  domain_adversarial_*_fixed.pkl (domain-adversarial, dlw=0.1).
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

import train_full_pie_scalenorm as _pie_mod
import train_full_jaad_scalenorm as _jaad_mod

from pie_data import PIE
from jaad_data import JAAD
from sf_gru_torch_scalenorm import ScaleNormSFGRUTorch


def load_result_model_paths(results_pkl):
    with open(results_pkl, 'rb') as f:
        d = pickle.load(f)
    return [(r['model_path'], r.get('best_threshold', 0.5)) for r in d['runs']]


def eval_cross(model_paths_thresholds, target_imdb, target_data_opts,
               source_name, target_name):
    if target_name == 'jaad':
        import sf_gru_torch as _sfgru_mod
        _sfgru_mod.SFGRUTorch.get_pose = _jaad_mod._safe_get_pose_jaad
        _sfgru_mod.SFGRUTorch.load_images_crop_and_process = _jaad_mod._load_images_crop_and_process_jaad
    else:
        import sf_gru_torch as _sfgru_mod
        _sfgru_mod.SFGRUTorch.get_pose = _pie_mod._safe_get_pose

    beh_test = target_imdb.generate_data_trajectory_sequence('test', **target_data_opts)

    results = []
    for seed_idx, (model_path, threshold) in enumerate(model_paths_thresholds):
        method = ScaleNormSFGRUTorch()
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
        os.path.join(RESULTS_DIR, 'full_pie_scalenorm_results_rtmpose.pkl'))
    jaad_model_paths = load_result_model_paths(
        os.path.join(RESULTS_DIR, 'full_jaad_scalenorm_results_rtmpose.pkl'))

    pie_imdb = PIE(data_path=PIE_DATA_DIR)
    jaad_imdb = JAAD(data_path=JAAD_DATA_DIR)

    log.info('=== PIE-trained (scale-norm box) models -> JAAD test set ===')
    pie_to_jaad = eval_cross(
        pie_model_paths, jaad_imdb, _jaad_mod.DATA_OPTS,
        source_name='pie', target_name='jaad')

    log.info('=== JAAD-trained (scale-norm box) models -> PIE test set ===')
    jaad_to_pie = eval_cross(
        jaad_model_paths, pie_imdb, _pie_mod.DATA_OPTS,
        source_name='jaad', target_name='pie')

    summary = {
        'pie_to_jaad': summarise(pie_to_jaad),
        'jaad_to_pie': summarise(jaad_to_pie),
    }

    out = os.path.join(RESULTS_DIR, 'cross_dataset_audit_scalenorm.pkl')
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
