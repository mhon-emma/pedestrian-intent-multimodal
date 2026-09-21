"""
cross_dataset_eval_attention.py
===================================
Generalizes cross_dataset_fusion_eval.py's pattern (plain cross-dataset
audit + modality-zeroing ablation, combined in one script) to the five
attention-family architectures in sf_gru_torch_attention.py that have
never been evaluated cross-dataset before:

  pose_box         PoseAttentionSFGRU(attention_on='box')
  pose_pose        PoseAttentionSFGRU(attention_on='pose')
  modality_fusion  ModalityFusionSFGRU
  cross_modal      CrossModalSFGRU
  other_modal      OtherModalSFGRU

Requires: train_full_pie_attention_nospeed.py and
train_full_jaad_attention_nospeed.py have already been run for the
requested --architecture (both directions' result pkls must exist).

Usage
-----
  python cross_dataset_eval_attention.py --architecture pose_box

Output
------
  results/cross_dataset_attention_audit_<architecture>.pkl
"""

import argparse
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

import train_full_pie_attention_nospeed as _pie_mod
import train_full_jaad_attention_nospeed_behonly as _jaad_mod

import sf_gru_torch as _sfgru_mod
from pie_data import PIE
from jaad_data import JAAD
from sf_gru_torch_attention import (CrossModalSFGRU, ModalityFusionSFGRU,
                                    OtherModalSFGRU, PoseAttentionSFGRU)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

ARCHITECTURES = {
    'pose_box':        lambda: PoseAttentionSFGRU(attention_on='box'),
    'pose_pose':       lambda: PoseAttentionSFGRU(attention_on='pose'),
    'modality_fusion': lambda: ModalityFusionSFGRU(),
    'cross_modal':     lambda: CrossModalSFGRU(),
    'other_modal':     lambda: OtherModalSFGRU(),
}

MODALITIES = ['local_box', 'local_context', 'pose', 'box']


def load_result_model_paths(results_pkl):
    with open(results_pkl, 'rb') as f:
        d = pickle.load(f)
    return [(r['model_path'], r.get('best_threshold', 0.5)) for r in d['runs']]


def apply_target_monkeypatches(target_name):
    """Re-applies BOTH get_pose and get_path for the target dataset.
    Importing both train_full_pie_attention_nospeed and
    train_full_jaad_attention_nospeed at module level means each
    import's get_path monkeypatch overwrites the previous one
    (last-imported wins) -- get_pose alone was being re-patched per
    direction here, but get_path was NOT, so the JAAD-trained -> PIE-test
    direction was silently resolving pose file paths into
    data/features/jaad/poses/ instead of data/features/pie/poses/,
    producing 100% pose-missing for that direction (caught live via a
    "Pose lookup: 8918/8918 frames missing (100.0%)" log line when
    running the pose_box architecture's JAAD->PIE audit, then confirmed
    by direct get_path() inspection). Fixed by re-patching get_path here
    too, same as train_domain_adversarial.py's
    apply_dataset_monkeypatches() already does correctly. The identical
    bug was also found and fixed in cross_dataset_fusion_eval.py --
    the existing cross_attn JAAD->PIE cross-dataset result was generated
    with 100% pose missing and needs to be rerun."""
    import utils as _u
    if target_name == 'jaad':
        _sfgru_mod.SFGRUTorch.get_pose = _jaad_mod._safe_get_pose_jaad
        _sfgru_mod.SFGRUTorch.load_images_crop_and_process = _jaad_mod._load_images_crop_and_process_jaad
        _sfgru_mod.get_path = _jaad_mod._patched_get_path
        _u.get_path = _jaad_mod._patched_get_path
    else:
        _sfgru_mod.SFGRUTorch.get_pose = _pie_mod._safe_get_pose
        _sfgru_mod.get_path = _pie_mod._patched_get_path
        _u.get_path = _pie_mod._patched_get_path


def test_with_zeroed_modality(method, data_test, model_path, zero_modality=None, threshold=0.5):
    """Same helper as cross_dataset_fusion_eval.py -- works for any
    SFGRUTorch subclass since build_model/get_data are inherited
    unchanged from the base class."""
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


def eval_direction(architecture, model_paths_thresholds, target_imdb,
                   target_data_opts, source_name, target_name):
    apply_target_monkeypatches(target_name)
    beh_test = target_imdb.generate_data_trajectory_sequence('test', **target_data_opts)

    conditions = [None] + MODALITIES
    results = {cond: [] for cond in conditions}

    for seed_idx, (model_path, threshold) in enumerate(model_paths_thresholds):
        method = ARCHITECTURES[architecture]()
        for cond in conditions:
            acc, auc, f1, prec, rec = test_with_zeroed_modality(
                method, beh_test, model_path, zero_modality=cond, threshold=threshold)
            label = cond if cond is not None else 'none'
            log.info('[%s] %s->%s seed=%d zeroed=%-14s threshold=%.2f Acc=%.4f AUC=%.4f F1=%.4f',
                     architecture, source_name, target_name, seed_idx, label, threshold, acc, auc, f1)
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture', required=True, choices=list(ARCHITECTURES.keys()))
    args = parser.parse_args()

    pie_model_paths = load_result_model_paths(
        os.path.join(RESULTS_DIR, f'full_pie_attention_nospeed_{args.architecture}_rtmpose.pkl'))
    jaad_model_paths = load_result_model_paths(
        os.path.join(RESULTS_DIR, f'full_jaad_attention_nospeed_behonly_{args.architecture}_rtmpose.pkl'))

    pie_imdb = PIE(data_path=PIE_DATA_DIR)
    jaad_imdb = JAAD(data_path=JAAD_DATA_DIR)

    log.info('=== [%s] PIE-trained -> JAAD test, modality ablation ===', args.architecture)
    pie_to_jaad = eval_direction(
        args.architecture, pie_model_paths, jaad_imdb, _jaad_mod.DATA_OPTS,
        'pie', 'jaad')

    log.info('=== [%s] JAAD-trained -> PIE test, modality ablation ===', args.architecture)
    jaad_to_pie = eval_direction(
        args.architecture, jaad_model_paths, pie_imdb, _pie_mod.DATA_OPTS,
        'jaad', 'pie')

    summary = {
        'pie_to_jaad': {str(cond): summarise(runs) for cond, runs in pie_to_jaad.items()},
        'jaad_to_pie': {str(cond): summarise(runs) for cond, runs in jaad_to_pie.items()},
    }

    out = os.path.join(RESULTS_DIR, f'cross_dataset_attention_audit_behonly_{args.architecture}.pkl')
    with open(out, 'wb') as f:
        pickle.dump({'pie_to_jaad_runs': pie_to_jaad, 'jaad_to_pie_runs': jaad_to_pie,
                    'summary': summary, 'architecture': args.architecture}, f)
    log.info('Saved: %s', out)

    print('\n' + '=' * 88)
    print(f'{"Direction":<14} {"Zeroed":<14} {"AUC":>8} {"+/-":>6} {"Acc":>8} {"+/-":>6} {"F1":>8}')
    print('-' * 88)
    for direction, cond_summaries in summary.items():
        for cond, s in cond_summaries.items():
            print(f'{direction:<14} {cond:<14} {s["auc_mean"]:>8.4f} {s["auc_std"]:>6.4f} '
                  f'{s["acc_mean"]:>8.4f} {s["acc_std"]:>6.4f} {s["f1_mean"]:>8.4f}')
    print('=' * 88)


if __name__ == '__main__':
    main()
