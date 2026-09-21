"""
eval_twoframe_best_ckpt.py
==============================
Full-val-set precision/recall diagnostic for the two-frame LoRA
best-checkpoint adapters (vlm_lora_finetune_twoframe.py) -- mirrors the
diagnostic run on the single-frame LoRA adapters earlier this session
(which found the intermediate 40-item val-sampling metric can pick a
checkpoint that looks strong on that slice but is actually close to
majority-class prediction on the FULL val set; JAAD's extreme class
imbalance, ~9-10% positive, makes this especially likely -- 0.90-0.925
raw accuracy on a 40-item slice is close to what an always-predict-NO
classifier would score on JAAD's true distribution).

Usage
-----
  python eval_twoframe_best_ckpt.py --dataset pie
  python eval_twoframe_best_ckpt.py --dataset jaad

Output: printed precision/recall/TP/FP/TN/FN breakdown, no results
file (this is a one-off diagnostic, not a tracked experiment result).
"""

import argparse
import os
import sys

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

SFGRU_DIR = '/usr1/home/mehon/emma_pedestrian-intent-multimodal'
sys.path.insert(0, SFGRU_DIR)
sys.path.append('/usr1/home/mehon/PIE/utilities')
sys.path.append('/usr1/home/mehon/JAAD')
os.chdir(SFGRU_DIR)
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

from vlm_lora_finetune_twoframe import (build_twoframe_track_index, ask_vlm_twoframe,
                                        extract_yes_no)
from vlm_zeroshot_test import full_frame_boxed

MODEL_ID = 'Qwen/Qwen2-VL-2B-Instruct'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=['pie', 'jaad'])
    parser.add_argument('--checkpoint', default='best', choices=['best', 'last'])
    args = parser.parse_args()

    suffix = '_best' if args.checkpoint == 'best' else ''
    adapter_dir = os.path.join(SFGRU_DIR, 'models', f'vlm_lora_twoframe_{args.dataset}{suffix}')

    if args.dataset == 'pie':
        import train_full_pie_nospeed as _mod
        from pie_data import PIE
        imdb = PIE(data_path='/usr1/home/mehon/data_root/pie')
        os.environ['PIE_POSE_BACKEND'] = 'rtmpose'
    else:
        import train_full_jaad_nospeed as _mod
        from jaad_data import JAAD
        imdb = JAAD(data_path='/usr1/home/mehon/JAAD')
        os.environ['JAAD_POSE_BACKEND'] = 'rtmpose'

    beh_val = imdb.generate_data_trajectory_sequence('val', **_mod.DATA_OPTS)
    val_items = build_twoframe_track_index(beh_val, _mod.MODEL_OPTS['time_to_event'])

    base = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map='cuda:0')
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=200 * 200, max_pixels=360 * 360)

    preds, labels = [], []
    for it in val_items:
        img_early = full_frame_boxed(it['img_path_early'], it['bbox_early'])
        img_late = full_frame_boxed(it['img_path_late'], it['bbox_late'])
        response = ask_vlm_twoframe(model, processor, img_early, img_late)
        verdict = extract_yes_no(response)
        pred = 1 if verdict == 'YES' else (0 if verdict == 'NO' else -1)
        preds.append(pred)
        labels.append(it['label'])

    valid = [(p, l) for p, l in zip(preds, labels) if p != -1]
    tp = sum(1 for p, l in valid if p == 1 and l == 1)
    fp = sum(1 for p, l in valid if p == 1 and l == 0)
    tn = sum(1 for p, l in valid if p == 0 and l == 0)
    fn = sum(1 for p, l in valid if p == 0 and l == 1)
    acc = (tp + tn) / len(valid) if valid else float('nan')
    prec = tp / (tp + fp) if (tp + fp) else float('nan')
    rec = tp / (tp + fn) if (tp + fn) else float('nan')
    n_pred_yes = sum(1 for p, l in valid if p == 1)

    print('\n' + '=' * 60)
    print(f'Two-frame LoRA [{args.checkpoint}] -- {args.dataset} full val set')
    print('-' * 60)
    print(f'n_valid={len(valid)}/{len(val_items)}')
    print(f'acc={acc:.4f} prec={prec:.4f} rec={rec:.4f}')
    print(f'TP={tp} FP={fp} TN={tn} FN={fn}')
    print(f'n_pred_YES={n_pred_yes}/{len(valid)} ({100*n_pred_yes/len(valid):.1f}%)')
    print(f'true positive rate in data: {sum(labels)/len(labels)*100:.1f}%')
    print('=' * 60)


if __name__ == '__main__':
    main()
