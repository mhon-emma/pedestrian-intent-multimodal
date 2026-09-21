"""
vlm_lora_finetune.py
========================
LoRA fine-tune of Qwen2-VL-2B-Instruct for PIE/JAAD crossing-intent
prediction, trained separately on each dataset's train split and
cross-tested the same way as every SF-GRU variant this project has
run (train on source, evaluate on the OTHER dataset's held-out test
split, never touching its train/val).

This directly extends the zero-shot probe (vlm_zeroshot_test.py) --
same full-frame-with-red-box image convention, same
obs_frame_idx = -time_to_event-1 windowing, same YES/NO framing -- but
now the model's weights are updated via LoRA on a VQA-style
(image, prompt) -> "YES"/"NO" objective instead of being queried
zero-shot.

Per the fine-tuning research (see project notes): full fine-tuning is
unnecessary and LoRA is the standard approach at this scale; class
imbalance is a real risk (JAAD ~9-10% positive, PIE ~20-28% positive)
so positives are oversampled during training construction, matching
this codebase's existing `balanced=True` convention for the SF-GRU
scripts rather than inventing a new imbalance-handling scheme.

Usage
-----
  python vlm_lora_finetune.py --dataset pie --epochs 3
  python vlm_lora_finetune.py --dataset jaad --epochs 3

Output
------
  models/vlm_lora_<dataset>/   (LoRA adapter weights)
  results/vlm_lora_train_<dataset>.pkl  (loss curve, config)
"""

import argparse
import logging
import os
import pickle
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw
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

from vlm_zeroshot_test import PROMPT, full_frame_boxed

MODEL_ID = 'Qwen/Qwen2-VL-2B-Instruct'


def build_track_index(beh_data, time_to_event, min_track_len=None):
    """Every usable (image_path, bbox, label) triple at the standard
    observation frame, across the WHOLE split (not a balanced subsample
    -- that's handled separately via oversampling in the Dataset)."""
    if min_track_len is None:
        min_track_len = time_to_event + 1
    obs_frame_idx = -time_to_event - 1
    items = []
    for i in range(len(beh_data['image'])):
        track_len = len(beh_data['image'][i])
        if track_len < min_track_len:
            continue
        img_path = beh_data['image'][i][obs_frame_idx]
        bbox = beh_data['bbox'][i][obs_frame_idx]
        label = int(beh_data['activities'][i][0][0])
        items.append({'img_path': img_path, 'bbox': bbox, 'label': label})
    return items


class VLMIntentDataset(Dataset):
    """Oversamples the minority (crossing) class to roughly 1:1, matching
    this codebase's `get_data_sequence_balance` convention elsewhere
    (flip-augmentation there; plain repetition here since a VLM's image
    encoder does not benefit from a hand-rolled flip the way a
    from-scratch CNN feature does, and Qwen2-VL was not trained with a
    flip-invariance objective we can rely on)."""

    def __init__(self, items, processor, balanced=True, seed=0):
        self.processor = processor
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
        image = full_frame_boxed(it['img_path'], it['bbox'])
        answer = 'YES' if it['label'] == 1 else 'NO'
        return {'image': image, 'answer': answer, 'label': it['label']}


def collate_and_tokenize(batch, processor, device):
    """Builds the full chat-formatted (prompt + answer) sequence and
    masks the prompt tokens out of the loss so only the answer tokens
    ('YES'/'NO' + EOS) contribute to the LoRA gradient -- standard
    instruction-tuning masking, otherwise the model would also be
    optimized to reproduce the (fixed, uninformative) prompt text.

    Prompt length (for masking) CANNOT be measured with a text-only
    tokenizer call: apply_chat_template's plain-text image placeholder is
    a single token, but the real image processor expands it to a
    resolution-dependent number of <|image_pad|> vision tokens (dozens to
    hundreds, depending on MAX_IMAGE_PIXELS) -- so a text-only prompt
    length badly undercounts the true prompt span in the full tokenized
    sequence. (An earlier version of this function did exactly that and
    silently left ~95% of the sequence -- image tokens, the full prompt
    text, and the chat-template scaffolding -- unmasked; the model was
    being trained to reconstruct the prompt itself, not to predict
    YES/NO, which is why loss stayed flat around 4.45 while val accuracy
    swung wildly between checkpoints: confirmed by decoding the unmasked
    label span and finding it started inside the image-pad block, not at
    the assistant's answer.)

    Instead, locate the true answer span directly in each sample's own
    real input_ids by finding the last occurrence of the
    '<|im_start|>assistant\n' marker's token ids -- everything from just
    after that marker to the end (answer + <|im_end|>) is the loss
    target; everything before and including the marker (image tokens,
    prompt text, chat scaffolding) is masked out."""
    images = [b['image'] for b in batch]
    answers = [b['answer'] for b in batch]

    full_texts = []
    for ans in answers:
        messages = [{'role': 'user', 'content': [
            {'type': 'image'}, {'type': 'text', 'text': PROMPT}]}]
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
        # Search from the end (marker is near the tail of the sequence,
        # right-padding puts real content first -- but search the whole
        # row to be robust to either padding side).
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


# Qwen2-VL's processor dynamically tokenizes images into a
# resolution-dependent number of vision tokens (28x28 patches). PIE/JAAD
# dashcam frames are ~1920x1080; left uncapped, a single full-resolution
# frame expands to thousands of image tokens, and the resulting
# attention/MLP activations (retained for backward through the LoRA
# adapters) are what actually exhausted 23.5GB, not batch size alone --
# confirmed by the first OOM occurring inside decoder_layer's MLP
# forward, not at model-load time. Capping to a max side length still
# leaves plenty of pixels for a VLM to read a red bounding box and its
# surrounding street context.
MAX_IMAGE_PIXELS = 512 * 512


def load_model_for_training(device='cuda:0'):
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from peft import LoraConfig, get_peft_model

    log.info('Loading %s for LoRA fine-tuning...', MODEL_ID)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map=device)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, min_pixels=256 * 256, max_pixels=MAX_IMAGE_PIXELS)

    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias='none',
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                        'gate_proj', 'up_proj', 'down_proj'],
        task_type='CAUSAL_LM',
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, processor


def evaluate(model, processor, items, device, max_items=None):
    """Greedy-decode YES/NO on held-out items (same extract_yes_no logic
    as zero-shot, imported to keep parsing consistent across both
    evaluation paths)."""
    from vlm_zeroshot_test import ask_vlm, extract_yes_no
    model.eval()
    if max_items is not None:
        items = items[:max_items]
    preds, labels = [], []
    for it in items:
        image = full_frame_boxed(it['img_path'], it['bbox'])
        response = ask_vlm(model, processor, image)
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

    train_items = build_track_index(beh_train, time_to_event)
    val_items = build_track_index(beh_val, time_to_event)
    log.info('%s: %d train items (pos rate %.3f), %d val items (pos rate %.3f)',
             args.dataset, len(train_items),
             np.mean([it['label'] for it in train_items]),
             len(val_items), np.mean([it['label'] for it in val_items]))

    model, processor = load_model_for_training(device=args.device)

    train_ds = VLMIntentDataset(train_items, processor, balanced=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=lambda b: b)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr)

    save_dir = os.path.join(MODELS_DIR, f'vlm_lora_{args.dataset}')
    best_dir = os.path.join(MODELS_DIR, f'vlm_lora_{args.dataset}_best')
    best_acc = -1.0
    best_step = None

    history = []
    step = 0
    micro_step = 0
    optimizer.zero_grad()
    for epoch in range(args.epochs):
        for batch in train_loader:
            inputs, labels = collate_and_tokenize(batch, processor, args.device)
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

                    # Small-data LoRA fine-tuning on this task converges fast and
                    # then overfits (confirmed empirically on the PIE run: val_acc
                    # peaked at step 300 then declined despite train loss going
                    # to ~0) -- so the LAST checkpoint is not necessarily the best
                    # one. Track and persist the best-by-val_acc checkpoint as
                    # training proceeds, not just whatever's left after the final
                    # step.
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

    out = os.path.join(RESULTS_DIR, f'vlm_lora_train_{args.dataset}.pkl')
    with open(out, 'wb') as f:
        pickle.dump({'history': history, 'final_val_eval': final_eval,
                    'best_val_acc': best_acc, 'best_step': best_step,
                    'args': vars(args), 'model_path': save_dir,
                    'best_model_path': best_dir if best_step is not None else None}, f)
    log.info('Saved: %s', out)


if __name__ == '__main__':
    main()
