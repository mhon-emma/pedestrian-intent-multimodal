"""
cross_dataset_eval_finetuned_context.py
==========================================
Targeted fix #2's actual test: does unfreezing VGG16's last conv block +
a small adapter for local_context (train_full_{pie,jaad}_finetuned_context.py)
close the PIE<->JAAD cross-dataset gap the way GRL+contrastive did
(baseline 0.498/0.402 -> GRL+contrastive 0.618/0.583)? In-domain training
alone doesn't answer this -- this script loads each direction's trained
checkpoint and evaluates on the OTHER dataset's held-out test split.

Batched evaluation (not cross_dataset_eval_attention_behonly.py's
whole-test-set-on-GPU pattern): raw 224x224x3 images make that memory-
risky here, same reasoning as train_full_{pie,jaad}_finetuned_context.py's
own batched train/val/test loops.

No model_opts.pkl dependency (train_full_{pie,jaad}_finetuned_context.py
don't save one) -- MODEL_OPTS is read directly from the training
modules instead, matching how those scripts already reference it.

Requires: train_full_pie_finetuned_context.py and
train_full_jaad_finetuned_context.py have already been run (both
directions' result pkls must exist).

Usage
-----
  python cross_dataset_eval_finetuned_context.py

Output
------
  results/cross_dataset_audit_finetuned_context.pkl
"""

import logging
import os
import pickle
import sys

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

SFGRU_DIR     = '/usr1/home/mehon/emma_pedestrian-intent-multimodal'
PIE_UTIL_DIR  = '/usr1/home/mehon/PIE/utilities'
PIE_DATA_DIR  = '/usr1/home/mehon/data_root/pie'
JAAD_UTIL_DIR = '/usr1/home/mehon/JAAD'
JAAD_DATA_DIR = '/usr1/home/mehon/JAAD'
RESULTS_DIR   = os.path.join(SFGRU_DIR, 'results')

sys.path.insert(0, SFGRU_DIR)
sys.path.append(PIE_UTIL_DIR)
sys.path.append(JAAD_UTIL_DIR)
os.chdir(SFGRU_DIR)

import train_full_pie_finetuned_context as _pie_mod
import train_full_jaad_finetuned_context as _jaad_mod
# The get_pose/get_path/load_images_crop_and_process patches themselves
# live in these two modules (train_full_{pie,jaad}_finetuned_context.py
# each import them as their OWN internal _pie_mod/_jaad_mod aliases) --
# importing them again directly here, under distinct names, avoids the
# confusing/fragile double-attribute-access of reaching through
# _pie_mod._pie_mod.
import train_full_pie_nospeed as _pie_patch_mod
import train_full_jaad_nospeed_behonly as _jaad_patch_mod

import sf_gru_torch as _sfgru_mod
import utils as _u
from pie_data import PIE
from jaad_data import JAAD
from sf_gru_torch_finetuned_context import (FineTunedContextStackedGRU,
                                            FineTunedContextSFGRUTorch,
                                            FineTunedContextSFGRUTorchJAAD)


def load_result_model_paths(results_pkl):
    with open(results_pkl, 'rb') as f:
        d = pickle.load(f)
    return [(r['model_path'], r.get('best_threshold', 0.5)) for r in d['runs']]


def apply_target_monkeypatches(target_name):
    """Re-applies BOTH get_pose and get_path for the target dataset --
    see cross_dataset_eval_attention_behonly.py's apply_target_monkeypatches
    docstring for the full history of the bug this avoids (last-imported
    module's get_path patch silently wins for BOTH directions unless
    re-applied per direction, right before use)."""
    if target_name == 'jaad':
        _sfgru_mod.SFGRUTorch.get_pose = _jaad_patch_mod._safe_get_pose_jaad
        _sfgru_mod.SFGRUTorch.load_images_crop_and_process = _jaad_patch_mod._load_images_crop_and_process_jaad
        _sfgru_mod.get_path = _jaad_patch_mod._patched_get_path
        _u.get_path = _jaad_patch_mod._patched_get_path
    else:
        _sfgru_mod.SFGRUTorch.get_pose = _pie_patch_mod._safe_get_pose
        _sfgru_mod.get_path = _pie_patch_mod._patched_get_path
        _u.get_path = _pie_patch_mod._patched_get_path


def _batched_eval(model, inputs_cpu, batch_size, device):
    model.eval()
    n = len(inputs_cpu[0])
    preds = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch = [x[start:start + batch_size].to(device) for x in inputs_cpu]
            preds.append(model(batch).squeeze(-1).cpu())
    return torch.cat(preds).numpy().reshape(-1, 1)


def eval_cross(model_paths_thresholds, target_imdb, target_data_opts, target_model_opts,
              method_class, source_name, target_name, batch_size=16):
    apply_target_monkeypatches(target_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    beh_test = target_imdb.generate_data_trajectory_sequence('test', **target_data_opts)

    method = method_class(device=device)
    test_data, data_types, _ = method.get_data({'test': beh_test}, dict(target_model_opts))
    test_inputs_cpu = [torch.from_numpy(np.asarray(x)).float() for x in test_data['test'][0]]
    labels = np.asarray(test_data['test'][1]).reshape(-1)

    results = []
    for seed_idx, (model_path, threshold) in enumerate(model_paths_thresholds):
        checkpoint = torch.load(os.path.join(model_path, 'model.pt'), map_location=device)
        model = FineTunedContextStackedGRU(checkpoint['data_types'], checkpoint['data_sizes'],
                                           256).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])

        preds = _batched_eval(model, test_inputs_cpu, batch_size, device)
        predictions = (preds >= threshold).astype(int)
        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, zero_division=0)
        prec = precision_score(labels, predictions, zero_division=0)
        rec = recall_score(labels, predictions, zero_division=0)
        auc = roc_auc_score(labels, preds)
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
        os.path.join(RESULTS_DIR, 'full_pie_finetuned_context_results_rtmpose.pkl'))
    jaad_model_paths = load_result_model_paths(
        os.path.join(RESULTS_DIR, 'full_jaad_finetuned_context_results_rtmpose.pkl'))

    pie_imdb = PIE(data_path=PIE_DATA_DIR)
    jaad_imdb = JAAD(data_path=JAAD_DATA_DIR)

    log.info('=== PIE-trained (fine-tuned context) models -> JAAD test set ===')
    pie_to_jaad = eval_cross(
        pie_model_paths, jaad_imdb, _jaad_mod.DATA_OPTS, _jaad_mod.MODEL_OPTS,
        FineTunedContextSFGRUTorchJAAD, source_name='pie', target_name='jaad')

    log.info('=== JAAD-trained (fine-tuned context) models -> PIE test set ===')
    jaad_to_pie = eval_cross(
        jaad_model_paths, pie_imdb, _pie_mod.DATA_OPTS, _pie_mod.MODEL_OPTS,
        FineTunedContextSFGRUTorch, source_name='jaad', target_name='pie')

    summary = {
        'pie_to_jaad': summarise(pie_to_jaad),
        'jaad_to_pie': summarise(jaad_to_pie),
    }

    out = os.path.join(RESULTS_DIR, 'cross_dataset_audit_finetuned_context.pkl')
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
