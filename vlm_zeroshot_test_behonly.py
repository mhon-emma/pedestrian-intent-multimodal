"""
vlm_zeroshot_test.py
=======================
Cheap first signal for whether a vision-language model's pretrained
world knowledge confers cross-dataset robustness that the from-scratch
SF-GRU family (base, CrossAttentionSFGRU, box-excluded,
domain-adversarial, scale-normalized, class-balance-reweighted -- all
four of which failed to fix PIE<->JAAD transfer) lacks. This is a
ZERO-SHOT test (no fine-tuning): Qwen2-VL-2B-Instruct is shown a single
cropped frame (pedestrian + surrounding context, same crop convention as
local_context elsewhere in this codebase) and asked a natural-language
crossing-intent question, with no PIE/JAAD-specific training at all.

This is deliberately NOT a fair comparison to the trained SF-GRU
models: it uses a single frame (no temporal sequence), no bounding-box
trajectory, no pose, and no dataset-specific calibration -- it is a
first, cheap probe of whether general visual-language pretraining alone
carries any signal for this task before investing in fine-tuning
infrastructure, which would be a much larger effort (new data-loading
path, no reuse of the existing sf_gru_torch.py pipeline).

Usage
-----
  python vlm_zeroshot_test.py --dataset pie --n_samples 30
  python vlm_zeroshot_test.py --dataset jaad --n_samples 30

Output
------
  results/vlm_zeroshot_<dataset>.pkl
"""

import argparse
import logging
import os
import pickle
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

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


MODEL_ID = 'Qwen/Qwen2-VL-2B-Instruct'

PROMPT = (
    "You are looking at a photo taken from a car's dashboard camera. "
    "There is a pedestrian highlighted with a red bounding box. "
    "Look carefully at their body position, posture, gaze direction, "
    "distance from the curb or roadway, and whether they are already in "
    "motion toward the street. Many pedestrians in dashcam photos are "
    "standing still, walking parallel to the road, or waiting -- not "
    "about to cross. Only answer YES if there is clear visual evidence "
    "they are about to step into the road. "
    "First, write one short sentence describing the specific visual "
    "evidence for your answer. Then, on a new line, write exactly one "
    "word: YES or NO."
)


def load_model():
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    log.info('Loading %s ...', MODEL_ID)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map='cuda:0')
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    return model, processor


def full_frame_boxed(img_path, bbox):
    """Show the VLM the FULL dashcam frame with the target pedestrian
    boxed in red, rather than an enlarged/squarified tight crop.

    The tight-crop convention used elsewhere in this codebase
    (local_context: enlarge + squarify + pad_resize to 224x224) was built
    for a frozen CNN feature extractor, which doesn't need a
    human/VLM-legible image -- and on PIE's low-light dashcam footage,
    that crop produces tiny, dark, heavily distorted images with visible
    tiling artifacts from pad_resize's repeat-padding (confirmed by
    direct visual inspection: see logs_vlm_smoketest.log's all-'NO'
    zero-shot result, traced to unusable input images, not a genuine
    absence of signal). The full frame preserves scene context (crosswalk
    position, other traffic, road geometry) that a VLM can actually
    reason about, and is not artificially darkened/distorted by the crop
    pipeline -- much closer to how a human would judge the same photo."""
    img = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    box = list(map(int, bbox[0:4]))
    draw.rectangle(box, outline=(255, 0, 0), width=max(3, img.size[0] // 300))
    return img


def ask_vlm(model, processor, image):
    messages = [{
        'role': 'user',
        'content': [
            {'type': 'image', 'image': image},
            {'type': 'text', 'text': PROMPT},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors='pt').to('cuda:0')
    with torch.no_grad():
        # room for one reasoning sentence + the YES/NO line -- 60 tokens
        # was verified too tight (many responses were cut off mid-reasoning,
        # before the explicit YES/NO line), confirmed by inspecting raw
        # responses in logs_vlm_jaad_v2.log.
        output_ids = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    trimmed = output_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
    return response


def extract_yes_no(response):
    """The prompt asks for reasoning followed by YES/NO on its own line.
    Scan lines in reverse so the final verdict wins over any earlier
    mention of "yes"/"no" inside the reasoning sentence itself (e.g.
    'not yet stepping off the curb' should not be parsed as a YES)."""
    for line in reversed(response.strip().splitlines()):
        line = line.strip().upper().strip('.').strip()
        if line in ('YES', 'NO'):
            return line
        if line.endswith('YES') or line.endswith('NO'):
            # tolerate a trailing "... Answer: YES" on the same line
            last_word = line.split()[-1].strip('.').strip()
            if last_word in ('YES', 'NO'):
                return last_word
    # fallback 1: whole-response WORD search (explicit YES/NO token
    # present somewhere, just not alone on its own line). Must be a real
    # word-boundary match, not a raw substring -- 'NO' is a substring of
    # ordinary words like 'now'/'know', and a naive `in` check would
    # misfire on those.
    import re
    words = set(re.findall(r"[A-Z']+", response.upper()))
    has_yes = 'YES' in words
    has_no = 'NO' in words
    if has_yes and not has_no:
        return 'YES'
    if has_no and not has_yes:
        return 'NO'

    # fallback 2: the response was truncated before reaching an explicit
    # YES/NO token (confirmed to happen with a too-tight max_new_tokens
    # budget -- see ask_vlm's docstring), leaving only the reasoning
    # sentence, e.g. "The pedestrian is about to step into the road."
    # Detect an explicit negation of "about to cross/step into the road"
    # first (must check before the affirmative phrase, since a negated
    # sentence contains the same affirmative phrase as a substring).
    lower = response.lower()
    negation_markers = ('not about to', 'does not appear', 'is not', 'no clear visual evidence',
                        'not yet', "doesn't appear", 'unlikely to')
    affirmative_markers = ('about to step into the road', 'about to cross',
                          'stepping into the road', 'stepping off the curb')
    if any(m in lower for m in negation_markers):
        return 'NO'
    if any(m in lower for m in affirmative_markers):
        return 'YES'
    return 'UNCLEAR'


def sample_balanced(beh_data, n_samples, seed=0):
    """Pick n_samples//2 positive and n_samples//2 negative examples,
    using each track's LAST observed frame (closest to the labeled event)
    and its bbox at that frame -- single-frame zero-shot probe."""
    rng = np.random.RandomState(seed)
    labels = np.array([a[0][0] for a in beh_data['activities']])
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    n_each = n_samples // 2
    chosen_pos = rng.choice(pos_idx, size=min(n_each, len(pos_idx)), replace=False)
    chosen_neg = rng.choice(neg_idx, size=min(n_each, len(neg_idx)), replace=False)
    chosen = np.concatenate([chosen_pos, chosen_neg])
    rng.shuffle(chosen)
    return chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=['pie', 'jaad'])
    parser.add_argument('--n_samples', type=int, default=30)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    if args.dataset == 'pie':
        import train_full_pie_nospeed as _mod
        from pie_data import PIE
        imdb = PIE(data_path=PIE_DATA_DIR)
        os.environ['PIE_POSE_BACKEND'] = 'rtmpose'
    else:
        import train_full_jaad_nospeed_behonly as _mod
        from jaad_data import JAAD
        imdb = JAAD(data_path=JAAD_DATA_DIR)
        os.environ['JAAD_POSE_BACKEND'] = 'rtmpose'

    beh_test = imdb.generate_data_trajectory_sequence('test', **_mod.DATA_OPTS)
    idx = sample_balanced(beh_test, args.n_samples, seed=args.seed)
    log.info('Sampled %d examples (%s), positive rate=%.2f',
             len(idx), args.dataset,
             np.mean([beh_test['activities'][i][0][0] for i in idx]))

    model, processor = load_model()

    # Match the rest of this codebase's observation-window convention
    # (sf_gru_torch.py: track[-obs_length-time_to_event : -time_to_event]):
    # the "last observed frame" is time_to_event frames before the track's
    # end, NOT the raw last frame -- using the raw last frame would show
    # the VLM a frame at or after the labeled crossing event itself.
    time_to_event = _mod.MODEL_OPTS['time_to_event']
    obs_frame_idx = -time_to_event - 1

    results = []
    for i in idx:
        track_len = len(beh_test['image'][i])
        if track_len < time_to_event + 1:
            log.warning('Sample %d track too short (%d frames), skipping', i, track_len)
            continue
        img_path = beh_test['image'][i][obs_frame_idx]
        bbox = beh_test['bbox'][i][obs_frame_idx]
        label = int(beh_test['activities'][i][0][0])

        try:
            image = full_frame_boxed(img_path, bbox)
            response = ask_vlm(model, processor, image)
            verdict = extract_yes_no(response)
            pred = 1 if verdict == 'YES' else (0 if verdict == 'NO' else -1)
        except Exception as e:
            log.warning('Sample %d failed: %s', i, e)
            response, pred = 'ERROR', -1

        results.append({'idx': int(i), 'label': label, 'raw_response': response, 'pred': pred})
        log.info('idx=%d label=%d pred=%d raw=%r', i, label, pred, response)

    valid = [r for r in results if r['pred'] != -1]
    if valid:
        correct = sum(1 for r in valid if r['pred'] == r['label'])
        acc = correct / len(valid)
        # confusion breakdown
        tp = sum(1 for r in valid if r['pred'] == 1 and r['label'] == 1)
        fp = sum(1 for r in valid if r['pred'] == 1 and r['label'] == 0)
        tn = sum(1 for r in valid if r['pred'] == 0 and r['label'] == 0)
        fn = sum(1 for r in valid if r['pred'] == 0 and r['label'] == 1)
        prec = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
        rec = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
    else:
        acc = prec = rec = float('nan')
        tp = fp = tn = fn = 0

    summary = {'dataset': args.dataset, 'n_total': len(results), 'n_valid': len(valid),
              'acc': acc, 'precision': prec, 'recall': rec,
              'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}

    suffix = '_behonly' if args.dataset == 'jaad' else ''
    out = os.path.join(RESULTS_DIR, f'vlm_zeroshot_{args.dataset}{suffix}.pkl')
    with open(out, 'wb') as f:
        pickle.dump({'results': results, 'summary': summary}, f)
    log.info('Saved: %s', out)

    print('\n' + '=' * 60)
    print(f'VLM zero-shot ({MODEL_ID}) on {args.dataset} test (n={len(valid)}/{len(results)} valid)')
    print('-' * 60)
    print(f'Acc={acc:.3f}  Prec={prec:.3f}  Rec={rec:.3f}')
    print(f'TP={tp} FP={fp} TN={tn} FN={fn}')
    print('=' * 60)


if __name__ == '__main__':
    main()
