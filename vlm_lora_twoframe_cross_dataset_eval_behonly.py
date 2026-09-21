"""
vlm_lora_twoframe_cross_dataset_eval.py
===========================================
Cross-dataset test for the two-frame LoRA-fine-tuned VLM -- evaluate
the PIE-fine-tuned adapter on JAAD's held-out TEST split and vice
versa. Two-frame analogue of vlm_lora_cross_dataset_eval.py.

Usage
-----
  python vlm_lora_twoframe_cross_dataset_eval.py --direction pie_to_jaad
  python vlm_lora_twoframe_cross_dataset_eval.py --direction jaad_to_pie
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

from vlm_lora_finetune_twoframe import build_twoframe_track_index, ask_vlm_twoframe, extract_yes_no
from vlm_zeroshot_test import full_frame_boxed

MODEL_ID = 'Qwen/Qwen2-VL-2B-Instruct'


def evaluate_full(model, processor, items):
    preds, labels = [], []
    for it in items:
        img_early = full_frame_boxed(it['img_path_early'], it['bbox_early'])
        img_late = full_frame_boxed(it['img_path_late'], it['bbox_late'])
        response = ask_vlm_twoframe(model, processor, img_early, img_late)
        verdict = extract_yes_no(response)
        pred = 1 if verdict == 'YES' else (0 if verdict == 'NO' else -1)
        preds.append(pred)
        labels.append(it['label'])
    return preds, labels


def summarize(preds, labels):
    valid = [(p, l) for p, l in zip(preds, labels) if p != -1]
    n_unclear = len(preds) - len(valid)
    tp = sum(1 for p, l in valid if p == 1 and l == 1)
    fp = sum(1 for p, l in valid if p == 1 and l == 0)
    tn = sum(1 for p, l in valid if p == 0 and l == 0)
    fn = sum(1 for p, l in valid if p == 0 and l == 1)
    acc = (tp + tn) / len(valid) if valid else float('nan')
    prec = tp / (tp + fp) if (tp + fp) else float('nan')
    rec = tp / (tp + fn) if (tp + fn) else float('nan')
    f1 = (2 * prec * rec / (prec + rec)
          if (prec + rec) and not np.isnan(prec) and not np.isnan(rec) and (prec + rec) > 0
          else float('nan'))
    return {'n_total': len(preds), 'n_valid': len(valid), 'n_unclear': n_unclear,
            'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1,
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--direction', required=True, choices=['pie_to_jaad', 'jaad_to_pie'])
    parser.add_argument('--use_best_checkpoint', action='store_true', default=True)
    args = parser.parse_args()

    suffix = '_best' if args.use_best_checkpoint else ''

    if args.direction == 'pie_to_jaad':
        adapter_dir = os.path.join(MODELS_DIR, f'vlm_lora_twoframe_pie{suffix}')
        import train_full_jaad_nospeed_behonly as _mod
        from jaad_data import JAAD
        imdb = JAAD(data_path=JAAD_DATA_DIR)
        os.environ['JAAD_POSE_BACKEND'] = 'rtmpose'
    else:
        adapter_dir = os.path.join(MODELS_DIR, f'vlm_lora_twoframe_behonly_jaad{suffix}')
        import train_full_pie_nospeed as _mod
        from pie_data import PIE
        imdb = PIE(data_path=PIE_DATA_DIR)
        os.environ['PIE_POSE_BACKEND'] = 'rtmpose'

    if not os.path.isdir(adapter_dir):
        log.error('Adapter directory not found: %s', adapter_dir)
        sys.exit(1)

    beh_test = imdb.generate_data_trajectory_sequence('test', **_mod.DATA_OPTS)
    test_items = build_twoframe_track_index(beh_test, _mod.MODEL_OPTS['time_to_event'])
    log.info('%s: %d test items (pos rate %.3f), adapter=%s',
             args.direction, len(test_items),
             np.mean([it['label'] for it in test_items]), adapter_dir)

    base = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map='cuda:0')
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=200 * 200, max_pixels=360 * 360)

    preds, labels = evaluate_full(model, processor, test_items)
    summary = summarize(preds, labels)

    log.info('%s: acc=%.4f prec=%.4f rec=%.4f f1=%.4f (n_valid=%d/%d, n_unclear=%d)',
             args.direction, summary['acc'], summary['prec'], summary['rec'], summary['f1'],
             summary['n_valid'], summary['n_total'], summary['n_unclear'])
    log.info('TP=%d FP=%d TN=%d FN=%d', summary['tp'], summary['fp'], summary['tn'], summary['fn'])

    out = os.path.join(RESULTS_DIR, f'vlm_lora_twoframe_cross_dataset_{args.direction}_behonly{suffix}.pkl')
    with open(out, 'wb') as f:
        pickle.dump({'preds': preds, 'labels': labels, 'summary': summary,
                    'direction': args.direction, 'adapter_dir': adapter_dir}, f)
    log.info('Saved: %s', out)

    print('\n' + '=' * 70)
    print(f'Two-frame VLM LoRA cross-dataset ({args.direction})')
    print('-' * 70)
    for k, v in summary.items():
        print(f'{k}: {v}')
    print('=' * 70)


if __name__ == '__main__':
    main()
