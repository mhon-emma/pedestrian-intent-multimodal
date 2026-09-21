"""
vlm_lora_finetune_twoframe.py
=================================
Two-frame extension of vlm_lora_finetune.py: instead of a single
observation-frame image, each training/eval sample shows the model TWO
frames (an earlier frame ~10 frames before the observation frame, and
the observation frame itself) so it can see whether the pedestrian is
ALREADY IN MOTION toward the road -- a cue a single static frame cannot
carry. This mirrors vlm_zeroshot_variants.py's `two_frame` zero-shot
probe (which only ran a small, unscaled smoke test, n=6), but now as a
real LoRA fine-tune rather than a zero-shot prompt, since the earlier
zero-shot single-frame vs multi-prompt-variant sweep this session found
prompt framing alone didn't help much -- motion information might need
to be learned into the weights rather than just described in the
prompt.

Reuses vlm_lora_finetune.py's data-loading conventions (build_track_index,
oversampling, threshold/masking logic) but everywhere a single image
was used, this version uses an (earlier_frame, observation_frame) pair.

Usage
-----
  python vlm_lora_finetune_twoframe.py --dataset pie --epochs 3
  python vlm_lora_finetune_twoframe.py --dataset jaad --epochs 3

Output
------
  models/vlm_lora_twoframe_<dataset>/       (LoRA adapter, last checkpoint)
  models/vlm_lora_twoframe_<dataset>_best/  (best intermediate-val checkpoint)
  results/vlm_lora_twoframe_train_<dataset>.pkl
"""

import argparse
import logging
import os
import pickle
import sys

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

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
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

sys.path.insert(0, SFGRU_DIR)
sys.path.append(PIE_UTIL_DIR)
sys.path.append(JAAD_UTIL_DIR)
os.chdir(SFGRU_DIR)

from vlm_zeroshot_test import full_frame_boxed
from vlm_lora_finetune import MAX_IMAGE_PIXELS

MODEL_ID = 'Qwen/Qwen2-VL-2B-Instruct'
FRAME_GAP = 10  # frames between the "earlier" and observation frame

TWO_FRAME_PROMPT = (
    "You are looking at two photos taken a fraction of a second apart "
    "from a car's dashboard camera, showing the same pedestrian "
    "highlighted with a red bounding box. The first photo is earlier; "
    "the second is more recent. "
    "Compare the pedestrian's position and posture between the two "
    "photos to judge whether they are already moving toward the road. "
    "Many pedestrians are standing still, walking parallel to the road, "
    "or waiting -- not about to cross. Only answer YES if there is "
    "clear visual evidence of motion toward the road between the two "
    "photos. Answer with exactly one word: YES or NO."
)


def build_twoframe_track_index(beh_data, time_to_event, frame_gap=FRAME_GAP):
    """Same as vlm_lora_finetune.build_track_index, but requires enough
    track length for BOTH the observation frame and one frame_gap
    frames earlier, and returns both image paths/bboxes per item."""
    min_track_len = time_to_event + frame_gap + 1
    obs_frame_idx = -time_to_event - 1
    earlier_frame_idx = obs_frame_idx - frame_gap
    items = []
    for i in range(len(beh_data['image'])):
        track_len = len(beh_data['image'][i])
        if track_len < min_track_len:
            continue
        items.append({
            'img_path_early': beh_data['image'][i][earlier_frame_idx],
            'bbox_early': beh_data['bbox'][i][earlier_frame_idx],
            'img_path_late': beh_data['image'][i][obs_frame_idx],
            'bbox_late': beh_data['bbox'][i][obs_frame_idx],
            'label': int(beh_data['activities'][i][0][0]),
        })
    return items


class VLMTwoFrameIntentDataset(Dataset):
    """Same oversampling convention as VLMIntentDataset (see
    vlm_lora_finetune.py) -- plain repetition of minority-class items to
    ~1:1, not flip-augmentation (a VLM's pretrained image encoder isn't
    known to be flip-invariant, and neither of these image "frames" is
    being flipped here)."""

    def __init__(self, items, balanced=True, seed=0):
        if balanced:
            rng = np.random.RandomState(seed)
            pos = [it for it in items if it['label'] == 1]
            neg = [it for it in items if it['label'] == 0]
            if pos and neg:
                n_target = max(len(pos), len(neg))
                pos_rep = [pos[i] for i in rng.randint(0, len(pos), n_target)]
                neg_rep = [neg[i] for i in rng.randint(0, len(neg), n_target)]
                items = pos_rep + neg_rep
                rng.shuffle(items)
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        img_early = full_frame_boxed(it['img_path_early'], it['bbox_early'])
        img_late = full_frame_boxed(it['img_path_late'], it['bbox_late'])
        answer = 'YES' if it['label'] == 1 else 'NO'
        return {'images': [img_early, img_late], 'answer': answer, 'label': it['label']}


def collate_and_tokenize_twoframe(batch, processor, device):
    """Two-image analogue of vlm_lora_finetune.collate_and_tokenize --
    see that function's docstring for why prompt-length masking must be
    done via the assistant-turn-marker search, not a text-only tokenizer
    call. The only structural difference here: each sample contributes
    TWO images to the flat `images` list (processor expects either one
    flat list matching total image-placeholder count across the batch,
    or a list-of-lists -- we pass list-of-lists, one 2-image list per
    sample, which is the documented multi-image-per-sample batching
    convention for Qwen2-VL's processor)."""
    images = [b['images'] for b in batch]  # list of [img_early, img_late] pairs
    answers = [b['answer'] for b in batch]

    full_texts = []
    for ans in answers:
        messages = [{'role': 'user', 'content': [
            {'type': 'image'}, {'type': 'image'}, {'type': 'text', 'text': TWO_FRAME_PROMPT}]}]
        prompt_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        full_texts.append(prompt_text + ans + processor.tokenizer.eos_token)

    inputs = processor(text=full_texts, images=images, return_tensors='pt', padding=True)

    assistant_marker_ids = processor.tokenizer(
        '<|im_start|>assistant\n', add_special_tokens=False)['input_ids']
    marker_len = len(assistant_marker_ids)
    marker_tensor = torch.tensor(assistant_marker_ids)

    labels = inputs['input_ids'].clone()
    labels[:, :] = -100
    for i in range(len(batch)):
        seq = inputs['input_ids'][i]
        answer_start = None
        for pos in range(len(seq) - marker_len, -1, -1):
            if torch.equal(seq[pos:pos + marker_len], marker_tensor):
                answer_start = pos + marker_len
                break
        if answer_start is None:
            raise RuntimeError(
                f'Could not locate assistant-turn marker in sample {i} -- '
                'tokenizer/template mismatch, aborting rather than silently '
                'training on a wrongly-masked sequence.')
        real_len = int(inputs['attention_mask'][i].sum())
        labels[i, answer_start:real_len] = seq[answer_start:real_len]
    labels[inputs['attention_mask'] == 0] = -100

    inputs = {k: v.to(device) for k, v in inputs.items()}
    labels = labels.to(device)
    return inputs, labels


def load_model_for_training(device='cuda:0'):
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from peft import LoraConfig, get_peft_model

    log.info('Loading %s for two-frame LoRA fine-tuning...', MODEL_ID)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map=device)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    # Two images per sample roughly doubles vision-token count vs the
    # single-frame script for the same per-image resolution cap -- cut
    # MAX_IMAGE_PIXELS accordingly (each frame smaller) to stay within
    # the same activation-memory budget that avoided the single-frame
    # script's original OOM (see vlm_lora_finetune.py's MAX_IMAGE_PIXELS
    # comment for the root-cause analysis this mirrors).
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, min_pixels=200 * 200, max_pixels=360 * 360)

    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias='none',
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                        'gate_proj', 'up_proj', 'down_proj'],
        task_type='CAUSAL_LM',
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, processor


def ask_vlm_twoframe(model, processor, img_early, img_late, device='cuda:0'):
    messages = [{'role': 'user', 'content': [
        {'type': 'image', 'image': img_early},
        {'type': 'image', 'image': img_late},
        {'type': 'text', 'text': TWO_FRAME_PROMPT}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img_early, img_late], return_tensors='pt').to(device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    trimmed = output_ids[:, inputs['input_ids'].shape[1]:]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


def extract_yes_no(response):
    """Same fallback structure as vlm_zeroshot_test.extract_yes_no --
    reused inline (not imported) since the two-frame prompt's answer
    format ('YES'/'NO' only, no reasoning sentence) makes the simpler
    subset of that function's logic sufficient."""
    import re
    for line in reversed(response.strip().splitlines()):
        line = line.strip().upper().strip('.').strip()
        if line in ('YES', 'NO'):
            return line
    words = set(re.findall(r"[A-Z']+", response.upper()))
    has_yes, has_no = 'YES' in words, 'NO' in words
    if has_yes and not has_no:
        return 'YES'
    if has_no and not has_yes:
        return 'NO'
    return 'UNCLEAR'


def evaluate(model, processor, items, device, max_items=None):
    model.eval()
    if max_items is not None:
        items = items[:max_items]
    preds, labels = [], []
    for it in items:
        img_early = full_frame_boxed(it['img_path_early'], it['bbox_early'])
        img_late = full_frame_boxed(it['img_path_late'], it['bbox_late'])
        response = ask_vlm_twoframe(model, processor, img_early, img_late, device)
        verdict = extract_yes_no(response)
        pred = 1 if verdict == 'YES' else (0 if verdict == 'NO' else -1)
        preds.append(pred)
        labels.append(it['label'])
    model.train()
    valid = [(p, l) for p, l in zip(preds, labels) if p != -1]
    if not valid:
        return {'acc': float('nan'), 'n_valid': 0, 'n_total': len(items)}
    acc = sum(1 for p, l in valid if p == l) / len(valid)
    return {'acc': acc, 'n_valid': len(valid), 'n_total': len(items)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=['pie', 'jaad'])
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--grad_accum_steps', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--eval_every_steps', type=int, default=100)
    parser.add_argument('--eval_n', type=int, default=40)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    if args.dataset == 'pie':
        import train_full_pie_nospeed as _mod
        from pie_data import PIE
        imdb = PIE(data_path=PIE_DATA_DIR)
        os.environ['PIE_POSE_BACKEND'] = 'rtmpose'
    else:
        import train_full_jaad_nospeed as _mod
        from jaad_data import JAAD
        imdb = JAAD(data_path=JAAD_DATA_DIR)
        os.environ['JAAD_POSE_BACKEND'] = 'rtmpose'

    time_to_event = _mod.MODEL_OPTS['time_to_event']

    beh_train = imdb.generate_data_trajectory_sequence('train', **_mod.DATA_OPTS)
    beh_val = imdb.generate_data_trajectory_sequence('val', **_mod.DATA_OPTS)

    train_items = build_twoframe_track_index(beh_train, time_to_event)
    val_items = build_twoframe_track_index(beh_val, time_to_event)
    log.info('%s: %d train items (pos rate %.3f), %d val items (pos rate %.3f)',
             args.dataset, len(train_items),
             np.mean([it['label'] for it in train_items]),
             len(val_items), np.mean([it['label'] for it in val_items]))

    model, processor = load_model_for_training(device=args.device)

    train_ds = VLMTwoFrameIntentDataset(train_items, balanced=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=lambda b: b)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr)

    save_dir = os.path.join(MODELS_DIR, f'vlm_lora_twoframe_{args.dataset}')
    best_dir = os.path.join(MODELS_DIR, f'vlm_lora_twoframe_{args.dataset}_best')
    best_acc = -1.0
    best_step = None

    history = []
    step = 0
    micro_step = 0
    optimizer.zero_grad()
    for epoch in range(args.epochs):
        for batch in train_loader:
            inputs, labels = collate_and_tokenize_twoframe(batch, processor, args.device)
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss / args.grad_accum_steps
            loss.backward()
            micro_step += 1

            if micro_step % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                optimizer.zero_grad()

                step += 1
                full_loss = float(loss.item()) * args.grad_accum_steps
                history.append({'step': step, 'epoch': epoch, 'loss': full_loss})
                if step % 20 == 0:
                    log.info('epoch=%d step=%d loss=%.4f', epoch, step, full_loss)

                if step % args.eval_every_steps == 0:
                    eval_metrics = evaluate(model, processor, val_items, args.device,
                                            max_items=args.eval_n)
                    log.info('epoch=%d step=%d val_acc=%.4f (n_valid=%d/%d)',
                             epoch, step, eval_metrics['acc'],
                             eval_metrics['n_valid'], eval_metrics['n_total'])
                    history.append({'step': step, 'epoch': epoch, 'val_eval': eval_metrics})

                    if eval_metrics['acc'] > best_acc:
                        best_acc = eval_metrics['acc']
                        best_step = step
                        model.save_pretrained(best_dir)
                        processor.save_pretrained(best_dir)
                        log.info('New best val_acc=%.4f at step=%d -> saved %s',
                                 best_acc, best_step, best_dir)

        log.info('=== epoch %d complete ===', epoch)

    final_eval = evaluate(model, processor, val_items, args.device, max_items=None)
    log.info('Final val (%s, full val set, LAST checkpoint): acc=%.4f (n_valid=%d/%d)',
             args.dataset, final_eval['acc'], final_eval['n_valid'], final_eval['n_total'])
    log.info('Best checkpoint by intermediate val sampling: step=%s acc=%.4f (dir=%s)',
             best_step, best_acc, best_dir)

    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)
    log.info('Saved LAST-checkpoint LoRA adapter: %s', save_dir)

    out = os.path.join(RESULTS_DIR, f'vlm_lora_twoframe_train_{args.dataset}.pkl')
    with open(out, 'wb') as f:
        pickle.dump({'history': history, 'final_val_eval': final_eval,
                    'best_val_acc': best_acc, 'best_step': best_step,
                    'args': vars(args), 'model_path': save_dir,
                    'best_model_path': best_dir if best_step is not None else None}, f)
    log.info('Saved: %s', out)


if __name__ == '__main__':
    main()
