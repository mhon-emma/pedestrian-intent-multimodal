"""
vlm_zeroshot_variants.py
===========================
Four follow-up probes on vlm_zeroshot_test.py's finding (Qwen2-VL-2B
zero-shot: weak, YES-biased classifier, ~consistent across PIE/JAAD
unlike every trained SF-GRU variant). Each variant tests a DIFFERENT
hypothesis for why the binary YES/NO probe was weak, rather than
re-tuning the same prompt:

  confidence   Ask for a 0-100 "will cross" score instead of YES/NO, then
               sweep a decision threshold post-hoc (same idea already
               validated for the trained models in
               cross_dataset_eval_retuned_threshold.py). Tests whether
               the model's RANKING is better than its default decision
               boundary suggests -- i.e. whether the YES-bias is a
               calibration problem, not an information problem.

  two_frame    Show two frames (the observation frame and one ~10 frames
               earlier) so the model can see whether the pedestrian is
               ALREADY IN MOTION toward the road -- a cue a single static
               frame cannot carry at all. Tests whether motion
               information, not just posture in one frame, is the
               missing signal.

  decomposed   Replace the single compound judgment ("will they cross?")
               with two concrete perceptual sub-questions asked
               separately (near a crosswalk/road? oriented or moving
               toward the street?), combined with a simple AND rule.
               Tests whether the compound question itself, not the
               model's perception, was the bottleneck.

  bigger_model Same confidence-score prompt as `confidence`, but with
               Qwen2-VL-7B-Instruct instead of 2B. Tests whether this is
               a capability ceiling rather than a task-framing problem.

All four reuse full_frame_boxed() and the observation-frame-index logic
from vlm_zeroshot_test.py; only the prompting/scoring strategy (and,
for two_frame, the image input) differs per variant.

Usage
-----
  python vlm_zeroshot_variants.py --variant confidence   --dataset pie  --n_samples 60
  python vlm_zeroshot_variants.py --variant two_frame     --dataset jaad --n_samples 60
  python vlm_zeroshot_variants.py --variant decomposed    --dataset pie  --n_samples 60
  python vlm_zeroshot_variants.py --variant bigger_model  --dataset jaad --n_samples 60

Output
------
  results/vlm_variant_<variant>_<dataset>.pkl
"""

import argparse
import logging
import os
import pickle
import re
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw
from sklearn.metrics import roc_auc_score

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

from vlm_zeroshot_test import full_frame_boxed, sample_balanced, extract_yes_no

MODEL_IDS = {
    'small': 'Qwen/Qwen2-VL-2B-Instruct',
    'big':   'Qwen/Qwen2-VL-7B-Instruct',
}

CONFIDENCE_PROMPT = (
    "You are looking at a photo taken from a car's dashboard camera. "
    "There is a pedestrian highlighted with a red bounding box. "
    "Look carefully at their body position, posture, gaze direction, "
    "distance from the curb or roadway, and whether they are already in "
    "motion toward the street. "
    "On a scale from 0 to 100, how likely is it that this pedestrian is "
    "about to step into the road? 0 means certainly not about to cross, "
    "100 means certainly about to cross. "
    "Answer with ONLY a single integer from 0 to 100, nothing else."
)

TWO_FRAME_PROMPT = (
    "You are looking at two photos taken a fraction of a second apart "
    "from a car's dashboard camera, showing the same pedestrian "
    "highlighted with a red bounding box. The first photo is earlier; "
    "the second is more recent. "
    "Compare the pedestrian's position and posture between the two "
    "photos to judge whether they are already moving toward the road. "
    "On a scale from 0 to 100, how likely is it that this pedestrian is "
    "about to step into the road? "
    "Answer with ONLY a single integer from 0 to 100, nothing else."
)

DECOMPOSED_PROMPT_1 = (
    "You are looking at a photo taken from a car's dashboard camera. "
    "There is a pedestrian highlighted with a red bounding box. "
    "Is this pedestrian within a few steps of the road, a curb, or a "
    "crosswalk (as opposed to being far from the road, e.g. against a "
    "building or in the middle of a sidewalk far from the street)? "
    "Answer with exactly one word: YES or NO."
)
DECOMPOSED_PROMPT_2 = (
    "You are looking at a photo taken from a car's dashboard camera. "
    "There is a pedestrian highlighted with a red bounding box. "
    "Is this pedestrian's body oriented toward the road, or already in "
    "a walking motion toward the road (as opposed to standing still "
    "facing away from the road, or walking parallel to it)? "
    "Answer with exactly one word: YES or NO."
)


def load_model(size='small'):
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    model_id = MODEL_IDS[size]
    log.info('Loading %s ...', model_id)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map='cuda:0')
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def generate(model, processor, images, prompt, max_new_tokens=16):
    content = [{'type': 'image', 'image': img} for img in images]
    content.append({'type': 'text', 'text': prompt})
    messages = [{'role': 'user', 'content': content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images, return_tensors='pt').to('cuda:0')
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = output_ids[:, inputs['input_ids'].shape[1]:]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


def extract_score(response):
    """Parse a 0-100 integer from the model's response. Two known
    response shapes require different handling, confirmed by direct
    testing:
      (a) "45 out of 100" / "45/100" -- the FIRST number is the answer,
          the second is the model restating the scale's upper bound.
      (b) "On a scale of 0 to 100, I'd say 45" -- the LAST number is the
          answer; the earlier "0" and "100" are the model echoing the
          prompt's own scale description.
    Pattern (a) is checked first since it is unambiguous; otherwise fall
    back to the last number, which is correct for (b) and for the
    common case of a single bare integer (where first == last anyway).
    Returns None if no plausible score is found (kept out of AUC
    computation, but reported so silent failures are visible rather than
    defaulting to a misleading 0 or 50)."""
    out_of_match = re.search(r'\b([0-9]{1,3})\s*(?:/|out of)\s*100\b', response, re.IGNORECASE)
    if out_of_match:
        val = int(out_of_match.group(1))
        if 0 <= val <= 100:
            return val

    matches = re.findall(r'\b(100|[0-9]{1,2})\b', response)
    if matches:
        val = int(matches[-1])
        if 0 <= val <= 100:
            return val
    return None


def get_dataset_handle(dataset):
    if dataset == 'pie':
        import train_full_pie_nospeed as _mod
        from pie_data import PIE
        imdb = PIE(data_path=PIE_DATA_DIR)
        os.environ['PIE_POSE_BACKEND'] = 'rtmpose'
    else:
        import train_full_jaad_nospeed as _mod
        from jaad_data import JAAD
        imdb = JAAD(data_path=JAAD_DATA_DIR)
        os.environ['JAAD_POSE_BACKEND'] = 'rtmpose'
    return imdb, _mod


def run_confidence(dataset, n_samples, seed, model_size='small'):
    imdb, _mod = get_dataset_handle(dataset)
    beh_test = imdb.generate_data_trajectory_sequence('test', **_mod.DATA_OPTS)
    idx = sample_balanced(beh_test, n_samples, seed=seed)
    time_to_event = _mod.MODEL_OPTS['time_to_event']
    obs_frame_idx = -time_to_event - 1

    model, processor = load_model(model_size)
    results = []
    for i in idx:
        if len(beh_test['image'][i]) < time_to_event + 1:
            continue
        img_path = beh_test['image'][i][obs_frame_idx]
        bbox = beh_test['bbox'][i][obs_frame_idx]
        label = int(beh_test['activities'][i][0][0])
        try:
            image = full_frame_boxed(img_path, bbox)
            response = generate(model, processor, [image], CONFIDENCE_PROMPT)
            score = extract_score(response)
        except Exception as e:
            log.warning('Sample %d failed: %s', i, e)
            response, score = 'ERROR', None
        results.append({'idx': int(i), 'label': label, 'raw_response': response, 'score': score})
        log.info('idx=%d label=%d score=%s raw=%r', i, label, score, response)
    return results


def run_two_frame(dataset, n_samples, seed, frame_gap=10):
    imdb, _mod = get_dataset_handle(dataset)
    beh_test = imdb.generate_data_trajectory_sequence('test', **_mod.DATA_OPTS)
    idx = sample_balanced(beh_test, n_samples, seed=seed)
    time_to_event = _mod.MODEL_OPTS['time_to_event']
    obs_frame_idx = -time_to_event - 1
    earlier_frame_idx = obs_frame_idx - frame_gap

    model, processor = load_model('small')
    results = []
    for i in idx:
        track_len = len(beh_test['image'][i])
        if track_len < time_to_event + frame_gap + 1:
            continue
        img_path_early = beh_test['image'][i][earlier_frame_idx]
        bbox_early = beh_test['bbox'][i][earlier_frame_idx]
        img_path_late = beh_test['image'][i][obs_frame_idx]
        bbox_late = beh_test['bbox'][i][obs_frame_idx]
        label = int(beh_test['activities'][i][0][0])
        try:
            img_early = full_frame_boxed(img_path_early, bbox_early)
            img_late = full_frame_boxed(img_path_late, bbox_late)
            response = generate(model, processor, [img_early, img_late], TWO_FRAME_PROMPT)
            score = extract_score(response)
        except Exception as e:
            log.warning('Sample %d failed: %s', i, e)
            response, score = 'ERROR', None
        results.append({'idx': int(i), 'label': label, 'raw_response': response, 'score': score})
        log.info('idx=%d label=%d score=%s raw=%r', i, label, score, response)
    return results


def run_decomposed(dataset, n_samples, seed):
    imdb, _mod = get_dataset_handle(dataset)
    beh_test = imdb.generate_data_trajectory_sequence('test', **_mod.DATA_OPTS)
    idx = sample_balanced(beh_test, n_samples, seed=seed)
    time_to_event = _mod.MODEL_OPTS['time_to_event']
    obs_frame_idx = -time_to_event - 1

    model, processor = load_model('small')
    results = []
    for i in idx:
        if len(beh_test['image'][i]) < time_to_event + 1:
            continue
        img_path = beh_test['image'][i][obs_frame_idx]
        bbox = beh_test['bbox'][i][obs_frame_idx]
        label = int(beh_test['activities'][i][0][0])
        try:
            image = full_frame_boxed(img_path, bbox)
            resp1 = generate(model, processor, [image], DECOMPOSED_PROMPT_1, max_new_tokens=8)
            resp2 = generate(model, processor, [image], DECOMPOSED_PROMPT_2, max_new_tokens=8)
            v1, v2 = extract_yes_no(resp1), extract_yes_no(resp2)
            if v1 == 'YES' and v2 == 'YES':
                pred = 1
            elif v1 == 'UNCLEAR' or v2 == 'UNCLEAR':
                pred = -1
            else:
                pred = 0
            response = f'near_road={v1!r} oriented_toward={v2!r}'
        except Exception as e:
            log.warning('Sample %d failed: %s', i, e)
            response, pred = 'ERROR', -1
        results.append({'idx': int(i), 'label': label, 'raw_response': response, 'pred': pred})
        log.info('idx=%d label=%d pred=%d raw=%r', i, label, pred, response)
    return results


def summarise_score_based(results):
    valid = [r for r in results if r.get('score') is not None]
    if len(valid) < 2:
        return {'n_valid': len(valid), 'n_total': len(results), 'auc': float('nan')}
    labels = np.array([r['label'] for r in valid])
    scores = np.array([r['score'] for r in valid])
    try:
        auc = roc_auc_score(labels, scores)
    except ValueError:
        auc = float('nan')  # e.g. all-one-class among valid samples

    # threshold sweep for best-F1 operating point, same spirit as
    # find_best_threshold elsewhere in this codebase
    best_f1, best_thr, best_acc = -1, 50, float('nan')
    for thr in range(0, 101, 5):
        preds = (scores >= thr).astype(int)
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
            best_acc = (preds == labels).mean()

    return {'n_valid': len(valid), 'n_total': len(results), 'auc': float(auc),
           'best_threshold': best_thr, 'best_f1': float(best_f1), 'acc_at_best_thr': float(best_acc)}


def summarise_pred_based(results):
    valid = [r for r in results if r['pred'] != -1]
    if not valid:
        return {'n_valid': 0, 'n_total': len(results), 'acc': float('nan')}
    correct = sum(1 for r in valid if r['pred'] == r['label'])
    tp = sum(1 for r in valid if r['pred'] == 1 and r['label'] == 1)
    fp = sum(1 for r in valid if r['pred'] == 1 and r['label'] == 0)
    tn = sum(1 for r in valid if r['pred'] == 0 and r['label'] == 0)
    fn = sum(1 for r in valid if r['pred'] == 0 and r['label'] == 1)
    prec = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
    rec = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
    return {'n_valid': len(valid), 'n_total': len(results), 'acc': correct / len(valid),
           'precision': prec, 'recall': rec, 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', required=True,
                        choices=['confidence', 'two_frame', 'decomposed', 'bigger_model'])
    parser.add_argument('--dataset', required=True, choices=['pie', 'jaad'])
    parser.add_argument('--n_samples', type=int, default=60)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    if args.variant == 'confidence':
        results = run_confidence(args.dataset, args.n_samples, args.seed, model_size='small')
        summary = summarise_score_based(results)
    elif args.variant == 'two_frame':
        results = run_two_frame(args.dataset, args.n_samples, args.seed)
        summary = summarise_score_based(results)
    elif args.variant == 'decomposed':
        results = run_decomposed(args.dataset, args.n_samples, args.seed)
        summary = summarise_pred_based(results)
    elif args.variant == 'bigger_model':
        results = run_confidence(args.dataset, args.n_samples, args.seed, model_size='big')
        summary = summarise_score_based(results)
    else:
        raise ValueError(args.variant)

    summary['variant'] = args.variant
    summary['dataset'] = args.dataset

    out = os.path.join(RESULTS_DIR, f'vlm_variant_{args.variant}_{args.dataset}.pkl')
    with open(out, 'wb') as f:
        pickle.dump({'results': results, 'summary': summary}, f)
    log.info('Saved: %s', out)

    print('\n' + '=' * 60)
    print(f'VLM variant={args.variant} dataset={args.dataset}')
    print('-' * 60)
    for k, v in summary.items():
        print(f'{k}: {v}')
    print('=' * 60)


if __name__ == '__main__':
    main()
