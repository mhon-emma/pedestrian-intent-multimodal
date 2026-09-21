"""
cross_dataset_modality_ablation.py
====================================
Follow-up to cross_dataset_eval.py: cross_dataset_eval.py showed that SF-GRU
(no-speed variant) collapses under domain shift -- PIE-trained -> JAAD test
AUC 0.41 (below chance), JAAD-trained -> PIE test AUC 0.57 (near chance) --
versus in-domain AUC of 0.88 / 0.84 respectively. This script isolates WHICH
modality (local_box, local_context, pose, box) is driving that collapse by
zeroing each one out, one at a time, during cross-dataset evaluation and
re-measuring AUC. If zeroing a modality *recovers* performance relative to
the all-modalities cross-eval, that modality's dataset-specific statistics
(e.g. VGG16 features that encode PIE's Toronto street scenes vs JAAD's
scenes, or a systematic bounding-box distribution shift) are the likely
cause of the collapse rather than genuine intention signal.

Requires: train_full_pie_nospeed.py and train_full_jaad_nospeed.py results
(same models used by cross_dataset_eval.py).

Usage
-----
  python cross_dataset_modality_ablation.py

Output
------
  results/cross_dataset_modality_ablation.pkl
  Prints a table of AUC per (direction, zeroed-modality).
"""

import logging
import os
import pickle
import sys

import numpy as np
import torch

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

import train_full_pie_nospeed as _pie_mod
import train_full_jaad_nospeed as _jaad_mod

import sf_gru_torch as _sfgru_mod
from pie_data import PIE
from jaad_data import JAAD
from sf_gru_torch import SFGRUTorch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

MODALITIES = ['local_box', 'local_context', 'pose', 'box']


def load_result_model_paths(results_pkl):
    """Returns (model_path, best_threshold) per seed."""
    with open(results_pkl, 'rb') as f:
        d = pickle.load(f)
    return [(r['model_path'], r.get('best_threshold', 0.5)) for r in d['runs']]


def test_with_zeroed_modality(method, data_test, model_path, zero_modality=None, threshold=0.5):
    """Reimplements SFGRUTorch.test() but optionally zeroes one modality's
    array before the forward pass, to isolate its contribution."""
    with open(os.path.join(model_path, 'model_opts.pkl'), 'rb') as fid:
        model_opts = pickle.load(fid)

    checkpoint = torch.load(os.path.join(model_path, 'model.pt'), map_location=method.device)
    model = method.build_model(checkpoint['data_types'], checkpoint['data_sizes'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    test_data, data_types, _ = method.get_data({'test': data_test}, model_opts)
    raw_inputs = test_data['test'][0]
    labels = np.asarray(test_data['test'][1])

    inputs = []
    for name, arr in zip(data_types, raw_inputs):
        arr = np.asarray(arr).astype(np.float32)
        if zero_modality is not None and name == zero_modality:
            arr = np.zeros_like(arr)
        inputs.append(torch.from_numpy(arr).float().to(method.device))

    with torch.no_grad():
        preds = model(inputs).squeeze(-1).cpu().numpy().reshape(-1, 1)

    predictions = (preds >= threshold).astype(int)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, zero_division=0)
    prec = precision_score(labels, predictions, zero_division=0)
    rec = recall_score(labels, predictions, zero_division=0)
    auc = roc_auc_score(labels, preds)
    return acc, auc, f1, prec, rec


def eval_direction(model_paths_thresholds, target_imdb, target_data_opts, source_name, target_name):
    if target_name == 'jaad':
        _sfgru_mod.SFGRUTorch.get_pose = _jaad_mod._safe_get_pose_jaad
        _sfgru_mod.SFGRUTorch.load_images_crop_and_process = _jaad_mod._load_images_crop_and_process_jaad
    else:
        _sfgru_mod.SFGRUTorch.get_pose = _pie_mod._safe_get_pose

    beh_test = target_imdb.generate_data_trajectory_sequence('test', **target_data_opts)

    conditions = [None] + MODALITIES  # None = no modality zeroed (baseline, matches cross_dataset_eval.py)
    results = {cond: [] for cond in conditions}

    for seed_idx, (model_path, threshold) in enumerate(model_paths_thresholds):
        method = SFGRUTorch()
        for cond in conditions:
            acc, auc, f1, prec, rec = test_with_zeroed_modality(
                method, beh_test, model_path, zero_modality=cond, threshold=threshold)
            label = cond if cond is not None else 'none'
            log.info('%s->%s seed=%d zeroed=%-14s threshold=%.2f Acc=%.4f AUC=%.4f F1=%.4f',
                     source_name, target_name, seed_idx, label, threshold, acc, auc, f1)
            results[cond].append({'seed': seed_idx, 'acc': acc, 'auc': auc, 'f1': f1,
                                  'prec': prec, 'rec': rec, 'threshold': threshold})
    return results


def summarise(runs):
    aucs = np.array([r['auc'] for r in runs])
    accs = np.array([r['acc'] for r in runs])
    f1s = np.array([r['f1'] for r in runs])
    return {'auc_mean': aucs.mean(), 'auc_std': aucs.std(),
            'acc_mean': accs.mean(), 'acc_std': accs.std(),
            'f1_mean': f1s.mean(), 'f1_std': f1s.std()}


def main():
    pie_model_paths = load_result_model_paths(
        os.path.join(RESULTS_DIR, 'full_pie_nospeed_results_rtmpose.pkl'))
    jaad_model_paths = load_result_model_paths(
        os.path.join(RESULTS_DIR, 'full_jaad_nospeed_results_rtmpose.pkl'))

    pie_imdb = PIE(data_path=PIE_DATA_DIR)
    jaad_imdb = JAAD(data_path=JAAD_DATA_DIR)

    log.info('=== PIE-trained -> JAAD test, modality ablation ===')
    pie_to_jaad = eval_direction(pie_model_paths, jaad_imdb, _jaad_mod.DATA_OPTS, 'pie', 'jaad')

    log.info('=== JAAD-trained -> PIE test, modality ablation ===')
    jaad_to_pie = eval_direction(jaad_model_paths, pie_imdb, _pie_mod.DATA_OPTS, 'jaad', 'pie')

    summary = {
        'pie_to_jaad': {str(cond): summarise(runs) for cond, runs in pie_to_jaad.items()},
        'jaad_to_pie': {str(cond): summarise(runs) for cond, runs in jaad_to_pie.items()},
    }

    out = os.path.join(RESULTS_DIR, 'cross_dataset_modality_ablation.pkl')
    with open(out, 'wb') as f:
        pickle.dump({'pie_to_jaad_runs': pie_to_jaad, 'jaad_to_pie_runs': jaad_to_pie,
                    'summary': summary}, f)
    log.info('Saved: %s', out)

    print('\n' + '=' * 78)
    print(f'{"Direction":<14} {"Zeroed":<14} {"AUC":>8} {"+/-":>6} {"Acc":>8} {"+/-":>6} {"F1":>8}')
    print('-' * 78)
    for direction, cond_summaries in summary.items():
        for cond, s in cond_summaries.items():
            print(f'{direction:<14} {cond:<14} {s["auc_mean"]:>8.4f} {s["auc_std"]:>6.4f} '
                  f'{s["acc_mean"]:>8.4f} {s["acc_std"]:>6.4f} {s["f1_mean"]:>8.4f}')
    print('=' * 78)


if __name__ == '__main__':
    main()
