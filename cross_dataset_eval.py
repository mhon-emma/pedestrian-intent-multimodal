"""
cross_dataset_eval.py
======================
Cross-dataset generalization audit for SF-GRU (no-speed variant).

Motivation: Gesnouin et al. 2022 ("Assessing Cross-dataset Generalization
of Pedestrian Crossing Predictors") showed that pedestrian crossing
predictors of that generation (SingleRNN, MultiRNN, SFRNN, PCPA, etc.)
generalize poorly -- sometimes below random-guess AUC -- when trained on
PIE and tested on JAAD, or vice versa. No paper we found in the 2024-2026
cross-modal fusion / lightweight-architecture literature (ADAPT,
TrajFusionNet, EfficientPIE, etc.) re-checks this; all report in-dataset
metrics only. This script re-runs that audit for SF-GRU base + attention +
fusion variants trained in this repo, using the no-speed models (see
train_full_pie_nospeed.py / train_full_jaad_nospeed.py) so a genuine
feature-semantics mismatch (PIE's continuous OBD speed vs. JAAD's discrete
vehicle_act, both stored in the same 'speed' slot) doesn't confound the
domain-shift measurement.

Requires: train_full_pie_nospeed.py and train_full_jaad_nospeed.py have
already been run (their per-seed 'model_path' is read back from the
results/full_{pie,jaad}_nospeed_results_rtmpose.pkl files they save).

Usage
-----
  python cross_dataset_eval.py

Output
------
  results/cross_dataset_audit.pkl
  Prints a PIE->JAAD and JAAD->PIE summary table.
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

# Import both nospeed training modules for their monkeypatches (get_pose,
# get_path, JAAD image loader) and data-generation configs -- but we
# reimplement the actual eval loop here instead of calling their main().
import train_full_pie_nospeed as _pie_mod
import train_full_jaad_nospeed as _jaad_mod

from pie_data import PIE
from jaad_data import JAAD
from sf_gru_torch import SFGRUTorch


def load_result_model_paths(results_pkl):
    """Returns (model_path, best_threshold) per seed. best_threshold falls
    back to 0.5 if a run has none (shouldn't happen now that both PIE and
    JAAD nospeed training call find_best_threshold on their own val set)."""
    with open(results_pkl, 'rb') as f:
        d = pickle.load(f)
    return [(r['model_path'], r.get('best_threshold', 0.5)) for r in d['runs']]


def eval_cross(model_paths_thresholds, target_imdb, target_data_opts, target_model_opts_dataset_label,
               source_name, target_name):
    """Load each seed's checkpoint (trained on source_name) and evaluate it
    on target_imdb's test split (target_name), using that seed's own
    val-tuned threshold from its source-dataset training run."""
    # generate_data_trajectory_sequence needs the *target* dataset's own
    # get_pose / image loader monkeypatch active, which depends on which
    # module (_pie_mod or _jaad_mod) was imported last -- both patch the
    # same sf_gru_torch.SFGRUTorch class attributes, so we re-apply the
    # correct one right before generating target data.
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
        method = SFGRUTorch()
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
        os.path.join(RESULTS_DIR, 'full_pie_nospeed_results_rtmpose.pkl'))
    jaad_model_paths = load_result_model_paths(
        os.path.join(RESULTS_DIR, 'full_jaad_nospeed_results_rtmpose.pkl'))

    pie_imdb = PIE(data_path=PIE_DATA_DIR)
    jaad_imdb = JAAD(data_path=JAAD_DATA_DIR)

    log.info('=== PIE-trained models -> JAAD test set ===')
    pie_to_jaad = eval_cross(
        pie_model_paths, jaad_imdb, _jaad_mod.DATA_OPTS, 'jaad_nospeed',
        source_name='pie', target_name='jaad')

    log.info('=== JAAD-trained models -> PIE test set ===')
    jaad_to_pie = eval_cross(
        jaad_model_paths, pie_imdb, _pie_mod.DATA_OPTS, 'pie_nospeed',
        source_name='jaad', target_name='pie')

    summary = {
        'pie_to_jaad': summarise(pie_to_jaad),
        'jaad_to_pie': summarise(jaad_to_pie),
    }

    out = os.path.join(RESULTS_DIR, 'cross_dataset_audit.pkl')
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
