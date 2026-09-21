"""
vlm_lora_cross_dataset_eval_topk_behonly.py
================================================
Checkpoint-selection robustness companion to
vlm_lora_cross_dataset_eval_behonly.py. That script evaluates ONE
checkpoint (last, or the single best-by-val_acc) on the target's
cross-dataset test split. For the JAAD-trained direction specifically,
"best" was chosen using a validation set of only 17 samples (JAAD's
sample_type='beh' val split -- a genuine, project-wide constraint, not
a bug in this script; see vlm_lora_finetune_behonly.py's docstring) --
a single flipped prediction moves val_acc by ~6 percentage points, so
picking one "best" checkpoint risks reporting a number that mostly
reflects which checkpoint got lucky on 17 samples rather than a robust
fine-tuning outcome.

This script instead evaluates ALL of vlm_lora_finetune_behonly.py's
saved top-K checkpoints (from 'top_k_checkpoints' in
results/vlm_lora_train_behonly_<dataset>.pkl) on the same cross-dataset
test split, and reports the mean/std/range of cross-dataset AUC-proxy
metrics (acc, prec, rec, f1 -- no true probability score is available
from greedy YES/NO decoding, see vlm_lora_cross_dataset_eval_behonly.py)
across checkpoints. A tight range across top-K checkpoints means the
single-best number from the companion script is trustworthy; a wide
range means it was checkpoint-selection noise and the paper should
report the distribution, not one point estimate.

Only meaningful for the jaad_to_pie direction (source=JAAD, the small
val split); PIE's val split (224 samples) does not have this problem,
so pie_to_jaad's single-checkpoint number from the companion script is
not re-litigated here, but can still be run for direct comparison if
--direction pie_to_jaad is passed.

Usage
-----
  python vlm_lora_cross_dataset_eval_topk_behonly.py --direction jaad_to_pie

Output
------
  results/vlm_lora_cross_dataset_topk_<direction>_behonly.pkl
"""

import argparse
import logging
import os
import pickle
import sys

import numpy as np
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

SFGRU_DIR     = '/usr1/home/mehon/emma_pedestrian-intent-multimodal'
PIE_UTIL_DIR  = '/usr1/home/mehon/PIE/utilities'
PIE_DATA_DIR  = '/usr1/home/mehon/data_root/pie'
JAAD_UTIL_DIR = '/usr1/home/mehon/JAAD'
JAAD_DATA_DIR = '/usr1/home/mehon/JAAD'
RESULTS_DIR   = os.path.join(SFGRU_DIR, 'results')
MODELS_DIR    = os.path.join(SFGRU_DIR, 'models')

sys.path.insert(0, SFGRU_DIR)
sys.path.append(PIE_UTIL_DIR)
sys.path.append(JAAD_UTIL_DIR)
os.chdir(SFGRU_DIR)

from vlm_lora_finetune_behonly import build_track_index, MAX_IMAGE_PIXELS
from vlm_zeroshot_test import full_frame_boxed, ask_vlm, extract_yes_no
from vlm_lora_cross_dataset_eval_behonly import (
    load_lora_model, evaluate_full, summarize, MODEL_ID)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--direction', required=True, choices=['pie_to_jaad', 'jaad_to_pie'])
    args = parser.parse_args()

    if args.direction == 'pie_to_jaad':
        train_pkl = os.path.join(RESULTS_DIR, 'vlm_lora_train_pie.pkl')
        import train_full_jaad_nospeed_behonly as _mod
        from jaad_data import JAAD
        imdb = JAAD(data_path=JAAD_DATA_DIR)
        os.environ['JAAD_POSE_BACKEND'] = 'rtmpose'
    else:
        train_pkl = os.path.join(RESULTS_DIR, 'vlm_lora_train_behonly_jaad.pkl')
        import train_full_pie_nospeed as _mod
        from pie_data import PIE
        imdb = PIE(data_path=PIE_DATA_DIR)
        os.environ['PIE_POSE_BACKEND'] = 'rtmpose'

    if not os.path.exists(train_pkl):
        log.error('%s not found -- run vlm_lora_finetune_behonly.py --dataset %s '
                  '--top_k 3 first (source side of this direction).',
                  train_pkl, 'jaad' if args.direction == 'pie_to_jaad' else 'pie')
        sys.exit(1)

    with open(train_pkl, 'rb') as f:
        train_result = pickle.load(f)
    top_k = train_result.get('top_k_checkpoints', [])
    if not top_k:
        log.error('%s has no top_k_checkpoints (trained before this robustness check was '
                  'added, or --top_k 1 was used) -- re-run vlm_lora_finetune_behonly.py '
                  'with --top_k 3.', train_pkl)
        sys.exit(1)

    beh_test = imdb.generate_data_trajectory_sequence('test', **_mod.DATA_OPTS)
    test_items = build_track_index(beh_test, _mod.MODEL_OPTS['time_to_event'])
    log.info('%s: %d test items (pos rate %.3f), %d checkpoints to evaluate',
             args.direction, len(test_items),
             np.mean([it['label'] for it in test_items]), len(top_k))

    per_checkpoint = []
    for ckpt in top_k:
        adapter_dir = ckpt['model_path']
        if not os.path.isdir(adapter_dir):
            log.warning('Checkpoint dir missing (skipping): %s', adapter_dir)
            continue
        log.info('=== rank=%d (train val_acc=%.4f, step=%d): %s ===',
                 ckpt['rank'], ckpt['val_acc'], ckpt['step'], adapter_dir)
        model, processor = load_lora_model(adapter_dir)
        preds, labels = evaluate_full(model, processor, test_items)
        summary = summarize(preds, labels)
        summary['rank'] = ckpt['rank']
        summary['train_val_acc'] = ckpt['val_acc']
        summary['train_step'] = ckpt['step']
        per_checkpoint.append(summary)
        log.info('  cross-dataset: acc=%.4f prec=%.4f rec=%.4f f1=%.4f',
                 summary['acc'], summary['prec'], summary['rec'], summary['f1'])
        del model
        torch.cuda.empty_cache()

    accs = [c['acc'] for c in per_checkpoint if not np.isnan(c['acc'])]
    f1s = [c['f1'] for c in per_checkpoint if not np.isnan(c['f1'])]
    recs = [c['rec'] for c in per_checkpoint if not np.isnan(c['rec'])]

    robustness_summary = {
        'acc_mean': float(np.mean(accs)) if accs else float('nan'),
        'acc_std': float(np.std(accs)) if accs else float('nan'),
        'acc_min': float(np.min(accs)) if accs else float('nan'),
        'acc_max': float(np.max(accs)) if accs else float('nan'),
        'f1_mean': float(np.mean(f1s)) if f1s else float('nan'),
        'f1_std': float(np.std(f1s)) if f1s else float('nan'),
        'rec_mean': float(np.mean(recs)) if recs else float('nan'),
        'rec_std': float(np.std(recs)) if recs else float('nan'),
        'n_checkpoints': len(per_checkpoint),
    }

    out = os.path.join(RESULTS_DIR, f'vlm_lora_cross_dataset_topk_{args.direction}_behonly.pkl')
    with open(out, 'wb') as f:
        pickle.dump({'direction': args.direction, 'per_checkpoint': per_checkpoint,
                    'robustness_summary': robustness_summary}, f)
    log.info('Saved: %s', out)

    print('\n' + '=' * 78)
    print(f'VLM LoRA cross-dataset checkpoint-selection robustness ({args.direction})')
    print('-' * 78)
    for c in per_checkpoint:
        print(f"  rank={c['rank']} train_val_acc={c['train_val_acc']:.4f} step={c['train_step']}  "
              f"-> cross-dataset acc={c['acc']:.4f} f1={c['f1']:.4f} rec={c['rec']:.4f}")
    print('-' * 78)
    print(f"acc:  {robustness_summary['acc_mean']:.4f} +/- {robustness_summary['acc_std']:.4f} "
          f"(range {robustness_summary['acc_min']:.4f}-{robustness_summary['acc_max']:.4f})")
    print(f"f1:   {robustness_summary['f1_mean']:.4f} +/- {robustness_summary['f1_std']:.4f}")
    print(f"rec:  {robustness_summary['rec_mean']:.4f} +/- {robustness_summary['rec_std']:.4f}")
    print('=' * 78)


if __name__ == '__main__':
    main()
