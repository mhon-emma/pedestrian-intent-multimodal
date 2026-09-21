"""
train_classbalance_fix.py
============================
Fourth proposed fix for the cross-dataset generalization failure,
targeted at a mechanism the other three (box exclusion, domain-
adversarial training, scale-normalized box) never addressed: PIE's
positive (crossing) rate is 20-28% across its splits while JAAD's is
consistently 9-10% -- a stable 2.5-3x difference in base rate (verified
directly against the data, see results.tex Section IV-G). Both datasets'
TRAINING splits are flip-augmented to a 50/50 balance by
get_data_sequence_balance (matching every other training script in this
paper), so the model never learns "positives are rare" during training;
cross_dataset_eval_retuned_threshold.py already showed that correcting
for this MISMATCH POST-HOC, by re-tuning only the decision threshold on
the target's own validation split, improves recall but does not fix the
underlying AUC collapse -- because AUC is threshold-independent, no
post-hoc threshold choice can fix a genuine ranking-quality problem.

This script instead intervenes at TRAINING TIME: FocalBCELoss's alpha
parameter (already present in sf_gru_torch.py's train(), unused by every
other script in this paper -- always left at the default 0.5, i.e.
positive and negative classes weighted equally) is set toward the
TARGET dataset's known low positive rate when training on the source,
biasing the model's learned decision boundary itself (not just where a
threshold is drawn on top of it) toward predicting positives as rare,
matching what it will actually see at target-test time.

This is speculative in one respect: it uses target-dataset knowledge
(its class balance) at source-training time, which is realistic in an
unsupervised-domain-adaptation sense (dataset-level statistics, not
labels, are commonly assumed available) but is a stronger assumption
than the other three fixes made.

Usage
-----
  python train_classbalance_fix.py --source pie --target jaad --seeds 3 --focal_alpha 0.3
  python train_classbalance_fix.py --source jaad --target pie --seeds 3 --focal_alpha 0.6

Output
------
  results/classbalance_fix_<source>_to_<target>_alpha<alpha>.pkl
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

SFGRU_DIR     = '/usr1/home/mehon/emma_pedestrian-intent-multimodal'
PIE_UTIL_DIR  = '/usr1/home/mehon/PIE/utilities'
PIE_DATA_DIR  = '/usr1/home/mehon/data_root/pie'
JAAD_UTIL_DIR = '/usr1/home/mehon/JAAD'
JAAD_DATA_DIR = '/usr1/home/mehon/JAAD'
RESULTS_DIR   = os.path.join(SFGRU_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

sys.path.insert(0, SFGRU_DIR)
sys.path.append(PIE_UTIL_DIR)
sys.path.append(JAAD_UTIL_DIR)
os.chdir(SFGRU_DIR)

import sf_gru_torch as _sfgru_mod
from pie_data import PIE
from jaad_data import JAAD
from sf_gru_torch import SFGRUTorch

import train_full_pie_nospeed as _pie_mod
import train_full_jaad_nospeed as _jaad_mod

MODEL_OPTS_BASE = {
    'obs_input_type': ['local_box', 'local_context', 'pose', 'box'],  # no speed
    'enlarge_ratio': 1.5,
    'pred_target_type': ['crossing'],
    'obs_length': 15,
    'time_to_event': 60,
    'normalize_boxes': True,
}


def apply_dataset_monkeypatches(dataset):
    """Same reasoning as train_domain_adversarial.py: train_full_pie_nospeed
    and train_full_jaad_nospeed each patch sf_gru_torch.get_pose / get_path
    at import time; since this script imports both, whichever was imported
    LAST silently wins for every subsequent call unless re-applied
    immediately before each dataset's own data generation / feature prep."""
    if dataset == 'pie':
        _sfgru_mod.SFGRUTorch.get_pose = _pie_mod._safe_get_pose
        _sfgru_mod.get_path = _pie_mod._patched_get_path
    elif dataset == 'jaad':
        _sfgru_mod.SFGRUTorch.get_pose = _jaad_mod._safe_get_pose_jaad
        _sfgru_mod.get_path = _jaad_mod._patched_get_path
        _sfgru_mod.SFGRUTorch.load_images_crop_and_process = _jaad_mod._load_images_crop_and_process_jaad
    else:
        raise ValueError(dataset)


def get_raw_data(dataset, split, pose_backend):
    apply_dataset_monkeypatches(dataset)
    if dataset == 'pie':
        imdb = PIE(data_path=PIE_DATA_DIR)
        os.environ['PIE_POSE_BACKEND'] = pose_backend
        return imdb.generate_data_trajectory_sequence(split, **_pie_mod.DATA_OPTS)
    elif dataset == 'jaad':
        imdb = JAAD(data_path=JAAD_DATA_DIR)
        os.environ['JAAD_POSE_BACKEND'] = pose_backend
        return imdb.generate_data_trajectory_sequence(split, **_jaad_mod.DATA_OPTS)
    raise ValueError(dataset)


def run_one_seed(seed, source, target, pose_backend, focal_alpha, epochs=100, lr=3e-5):
    torch.manual_seed(seed)
    np.random.seed(seed)

    log.info('=== seed=%d [%s->%s] focal_alpha=%.2f ===', seed, source, target, focal_alpha)

    src_train_raw = get_raw_data(source, 'train', pose_backend)
    src_val_raw   = get_raw_data(source, 'val', pose_backend)

    model_opts = dict(MODEL_OPTS_BASE)
    model_opts['dataset'] = f'{source}_classbalance_fix'

    apply_dataset_monkeypatches(source)  # get_data's internal get_pose calls need this active
    method = SFGRUTorch()

    saved_files_path = method.train(
        src_train_raw,
        data_val=src_val_raw,
        batch_size=32,
        epochs=epochs,
        lr=lr,
        model_opts=model_opts,
        focal_alpha=focal_alpha,
    )

    best_threshold = method.find_best_threshold(src_val_raw, saved_files_path, model_opts=model_opts)

    # Evaluate on TARGET's held-out test split -- the actual cross-dataset
    # transfer measurement, same protocol as every other script in this paper.
    apply_dataset_monkeypatches(target)
    tgt_test_raw = get_raw_data(target, 'test', pose_backend)
    tgt_model_opts = dict(MODEL_OPTS_BASE)
    tgt_model_opts['dataset'] = f'{target}_classbalance_fix_eval'
    acc, auc, f1, prec, rec = method.test(tgt_test_raw, saved_files_path, threshold=best_threshold)

    metrics = {'seed': seed, 'source': source, 'target': target, 'focal_alpha': focal_alpha,
              'acc': acc, 'auc': auc, 'f1': f1, 'prec': prec, 'rec': rec,
              'best_threshold': best_threshold, 'model_path': saved_files_path}
    log.info('seed=%d [%s->%s] alpha=%.2f  target-test Acc=%.4f AUC=%.4f F1=%.4f',
             seed, source, target, focal_alpha, acc, auc, f1)
    return metrics


def summarise(runs):
    aucs = np.array([r['auc'] for r in runs])
    accs = np.array([r['acc'] for r in runs])
    f1s  = np.array([r['f1']  for r in runs])
    return {
        'auc_mean': float(aucs.mean()), 'auc_std': float(aucs.std()),
        'acc_mean': float(accs.mean()), 'acc_std': float(accs.std()),
        'f1_mean':  float(f1s.mean()),  'f1_std':  float(f1s.std()),
        'n': len(runs),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, choices=['pie', 'jaad'])
    parser.add_argument('--target', required=True, choices=['pie', 'jaad'])
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--backend', default='rtmpose', choices=['rtmpose'])
    parser.add_argument('--focal_alpha', type=float, required=True,
                        help='FocalBCELoss alpha, weighting the positive '
                             '(crossing) class. Default training uses 0.5 '
                             '(balanced). Set below 0.5 when the target '
                             'dataset has a lower positive rate than the '
                             'source (e.g. PIE->JAAD), above 0.5 for the '
                             'reverse (JAAD->PIE).')
    args = parser.parse_args()
    if args.source == args.target:
        raise ValueError('--source and --target must differ')

    runs = []
    for seed in range(args.seeds):
        m = run_one_seed(seed, args.source, args.target, args.backend, args.focal_alpha)
        runs.append(m)

    summary = summarise(runs)
    summary.update({'source': args.source, 'target': args.target, 'focal_alpha': args.focal_alpha})

    alpha_tag = f'alpha{args.focal_alpha:g}'
    out = os.path.join(RESULTS_DIR, f'classbalance_fix_{args.source}_to_{args.target}_{alpha_tag}.pkl')
    with open(out, 'wb') as f:
        pickle.dump({'runs': runs, 'summary': summary}, f)
    log.info('Saved: %s', out)

    print('\n' + '=' * 78)
    print(f'Class-balance fix: {args.source} (focal_alpha={args.focal_alpha}) -> {args.target} test')
    print('-' * 78)
    print(f'{args.target} test AUC: {summary["auc_mean"]:.4f} +/- {summary["auc_std"]:.4f}')
    print(f'{args.target} test Acc: {summary["acc_mean"]:.4f} +/- {summary["acc_std"]:.4f}')
    print(f'{args.target} test F1:  {summary["f1_mean"]:.4f} +/- {summary["f1_std"]:.4f}')
    print('=' * 78)


if __name__ == '__main__':
    main()
