"""
cross_dataset_eval_retuned_threshold.py
==========================================
Rechecks the cross-dataset audit with the classification threshold
re-tuned on the TARGET dataset's own val split, instead of reusing the
threshold tuned on the SOURCE dataset's val split (what
cross_dataset_eval.py / cross_dataset_eval_noboxspeed.py do).

Motivation: PIE train/val positive rate is ~25%, JAAD's is ~20%, and
JAAD's test set specifically is 82.5%/17.5% negative/positive -- a
meaningfully different base rate than PIE. A threshold tuned on PIE's val
set and applied unchanged to JAAD's test set conflates two different
failure modes:
  (a) genuine feature-transfer failure (the model's learned decision
      boundary doesn't separate crossing/non-crossing on the target
      domain's *feature distribution*), vs.
  (b) miscalibration (the boundary is fine, but the threshold that
      worked for the source domain's base rate doesn't fit the target
      domain's base rate).
This script re-tunes the threshold using ONLY the target dataset's val
set (never its test set) before evaluating on the target test set, which
isolates (a) from (b). If AUC (threshold-independent) and F1 are still
poor after retuning, the failure is genuinely about feature transfer, not
calibration -- strengthening the paper's core claim rather than leaving
it vulnerable to a "you just used the wrong threshold" critique.

Runs both the with-box and no-box/speed checkpoint sets so the retuned
numbers are directly comparable to cross_dataset_eval.py /
cross_dataset_eval_noboxspeed.py.

Requires: train_full_pie_nospeed.py, train_full_jaad_nospeed.py,
train_full_pie_noboxspeed.py, train_full_jaad_noboxspeed.py have all
already been run (this script only reads their saved checkpoints).

Usage
-----
  python cross_dataset_eval_retuned_threshold.py

Output
------
  results/cross_dataset_audit_retuned.pkl
  Prints a table for both the with-box and no-box variants, each
  direction, comparing source-threshold vs. target-retuned-threshold.
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

import train_full_pie_nospeed as _pie_mod
import train_full_jaad_nospeed as _jaad_mod
import train_full_pie_noboxspeed as _pie_nb_mod
import train_full_jaad_noboxspeed as _jaad_nb_mod

import sf_gru_torch as _sfgru_mod
from pie_data import PIE
from jaad_data import JAAD
from sf_gru_torch import SFGRUTorch


def load_result_model_paths(results_pkl):
    with open(results_pkl, 'rb') as f:
        d = pickle.load(f)
    return [(r['model_path'], r.get('best_threshold', 0.5)) for r in d['runs']]


def apply_target_monkeypatches(target_name, variant_mod_pie, variant_mod_jaad):
    if target_name == 'jaad':
        _sfgru_mod.SFGRUTorch.get_pose = variant_mod_jaad._safe_get_pose_jaad
        _sfgru_mod.SFGRUTorch.load_images_crop_and_process = variant_mod_jaad._load_images_crop_and_process_jaad
    else:
        _sfgru_mod.SFGRUTorch.get_pose = variant_mod_pie._safe_get_pose


def eval_direction(model_paths_thresholds, target_imdb, target_data_opts,
                   target_model_opts, source_name, target_name,
                   variant_mod_pie, variant_mod_jaad, variant_label):
    apply_target_monkeypatches(target_name, variant_mod_pie, variant_mod_jaad)

    beh_val  = target_imdb.generate_data_trajectory_sequence('val', **target_data_opts)
    beh_test = target_imdb.generate_data_trajectory_sequence('test', **target_data_opts)

    results = []
    for seed_idx, (model_path, source_threshold) in enumerate(model_paths_thresholds):
        method = SFGRUTorch()

        # Re-tune threshold on the TARGET's own val set (never its test set).
        retuned_threshold = method.find_best_threshold(
            beh_val, model_path, model_opts=target_model_opts)

        acc_src, auc_src, f1_src, prec_src, rec_src = method.test(
            beh_test, model_path, threshold=source_threshold)
        acc_rt, auc_rt, f1_rt, prec_rt, rec_rt = method.test(
            beh_test, model_path, threshold=retuned_threshold)

        log.info('[%s] %s->%s seed=%d  src_thr=%.2f (Acc=%.4f F1=%.4f)  '
                 'retuned_thr=%.2f (Acc=%.4f F1=%.4f)  AUC=%.4f (unchanged by threshold)',
                 variant_label, source_name, target_name, seed_idx,
                 source_threshold, acc_src, f1_src,
                 retuned_threshold, acc_rt, f1_rt, auc_src)

        results.append({
            'seed': seed_idx, 'model_path': model_path,
            'source_threshold': source_threshold,
            'acc_source_thr': acc_src, 'f1_source_thr': f1_src,
            'prec_source_thr': prec_src, 'rec_source_thr': rec_src,
            'retuned_threshold': retuned_threshold,
            'acc_retuned_thr': acc_rt, 'f1_retuned_thr': f1_rt,
            'prec_retuned_thr': prec_rt, 'rec_retuned_thr': rec_rt,
            'auc': auc_src,  # threshold-independent; auc_rt is identical
        })
    return results


def summarise(runs, thr_kind):
    aucs = np.array([r['auc'] for r in runs])
    accs = np.array([r[f'acc_{thr_kind}_thr'] for r in runs])
    f1s  = np.array([r[f'f1_{thr_kind}_thr']  for r in runs])
    recs = np.array([r[f'rec_{thr_kind}_thr'] for r in runs])
    return {
        'auc_mean': aucs.mean(), 'auc_std': aucs.std(),
        'acc_mean': accs.mean(), 'acc_std': accs.std(),
        'f1_mean':  f1s.mean(),  'f1_std':  f1s.std(),
        'rec_mean': recs.mean(), 'rec_std': recs.std(),
        'n': len(runs),
    }


def run_variant(variant_label, pie_results_pkl, jaad_results_pkl,
                variant_mod_pie, variant_mod_jaad, pie_imdb, jaad_imdb):
    pie_model_paths = load_result_model_paths(os.path.join(RESULTS_DIR, pie_results_pkl))
    jaad_model_paths = load_result_model_paths(os.path.join(RESULTS_DIR, jaad_results_pkl))

    log.info('=== [%s] PIE-trained -> JAAD test (threshold retune) ===', variant_label)
    pie_to_jaad = eval_direction(
        pie_model_paths, jaad_imdb, variant_mod_jaad.DATA_OPTS, variant_mod_jaad.MODEL_OPTS,
        'pie', 'jaad', variant_mod_pie, variant_mod_jaad, variant_label)

    log.info('=== [%s] JAAD-trained -> PIE test (threshold retune) ===', variant_label)
    jaad_to_pie = eval_direction(
        jaad_model_paths, pie_imdb, variant_mod_pie.DATA_OPTS, variant_mod_pie.MODEL_OPTS,
        'jaad', 'pie', variant_mod_pie, variant_mod_jaad, variant_label)

    return {
        'pie_to_jaad_runs': pie_to_jaad,
        'jaad_to_pie_runs': jaad_to_pie,
        'summary': {
            'pie_to_jaad_source_thr':  summarise(pie_to_jaad, 'source'),
            'pie_to_jaad_retuned_thr': summarise(pie_to_jaad, 'retuned'),
            'jaad_to_pie_source_thr':  summarise(jaad_to_pie, 'source'),
            'jaad_to_pie_retuned_thr': summarise(jaad_to_pie, 'retuned'),
        }
    }


def main():
    pie_imdb = PIE(data_path=PIE_DATA_DIR)
    jaad_imdb = JAAD(data_path=JAAD_DATA_DIR)

    with_box = run_variant(
        'with_box', 'full_pie_nospeed_results_rtmpose.pkl', 'full_jaad_nospeed_results_rtmpose.pkl',
        _pie_mod, _jaad_mod, pie_imdb, jaad_imdb)

    no_box = run_variant(
        'no_box', 'full_pie_noboxspeed_results_rtmpose.pkl', 'full_jaad_noboxspeed_results_rtmpose.pkl',
        _pie_nb_mod, _jaad_nb_mod, pie_imdb, jaad_imdb)

    out = os.path.join(RESULTS_DIR, 'cross_dataset_audit_retuned.pkl')
    with open(out, 'wb') as f:
        pickle.dump({'with_box': with_box, 'no_box': no_box}, f)
    log.info('Saved: %s', out)

    print('\n' + '=' * 100)
    print(f'{"Variant":<10} {"Direction":<14} {"ThrSrc":<10} {"AUC":>8} {"+/-":>6} '
          f'{"Acc":>8} {"+/-":>6} {"F1":>8} {"Rec":>8}')
    print('-' * 100)
    for variant_label, data in [('with_box', with_box), ('no_box', no_box)]:
        for direction in ['pie_to_jaad', 'jaad_to_pie']:
            for thr_kind, thr_label in [('source_thr', 'source'), ('retuned_thr', 'retuned')]:
                s = data['summary'][f'{direction}_{thr_kind}']
                print(f'{variant_label:<10} {direction:<14} {thr_label:<10} '
                      f'{s["auc_mean"]:>8.4f} {s["auc_std"]:>6.4f} '
                      f'{s["acc_mean"]:>8.4f} {s["acc_std"]:>6.4f} '
                      f'{s["f1_mean"]:>8.4f} {s["rec_mean"]:>8.4f}')
    print('=' * 100)


if __name__ == '__main__':
    main()
