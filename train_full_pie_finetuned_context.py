"""
train_full_pie_finetuned_context.py
======================================
Targeted fix #2, PIE side. See sf_gru_torch_finetuned_context.py for the
full rationale: local_context features come from a frozen, never-fine-
tuned VGG16 -- the leakage probe's strongest dataset-identity signal of
any modality. This unfreezes VGG's last conv block (block5) + a small
adapter and trains it jointly with the crossing-intent objective,
in-domain first (no cross-dataset/GRL machinery -- isolating the
fine-tuning effect per this session's plan).

Custom train/test loop (NOT sf_gru_torch.py's SFGRUTorch.train/test):
those load the ENTIRE val/test set onto GPU at once, which is fine for
small precomputed feature vectors but risky for raw 224x224x3 images
(PIE's val set alone would be ~6.5GB resident for the whole run). This
script batches val/test evaluation too, keeping peak GPU memory bounded
regardless of dataset size.

Usage
-----
  python train_full_pie_finetuned_context.py --seeds 3

Output
------
  results/full_pie_finetuned_context_results_rtmpose.pkl
"""

import argparse
import logging
import os
import pickle
import sys

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

SFGRU_DIR    = '/usr1/home/mehon/emma_pedestrian-intent-multimodal'
PIE_UTIL_DIR = '/usr1/home/mehon/PIE/utilities'
PIE_DATA_DIR = '/usr1/home/mehon/data_root/pie'
RESULTS_DIR  = os.path.join(SFGRU_DIR, 'results')

sys.path.insert(0, SFGRU_DIR)
sys.path.append(PIE_UTIL_DIR)
os.chdir(SFGRU_DIR)

import sf_gru_torch as _sfgru_mod
from pie_data import PIE
from sf_gru_torch import FocalBCELoss
from sf_gru_torch_finetuned_context import FineTunedContextSFGRUTorch

# Reuse train_full_pie_nospeed.py's own get_pose (backend-aware pose
# loading, base SFGRUTorch.get_pose can't handle the rtmpose/openpose
# split) and get_path (redirects pose lookups to the shared
# data/features/pie/poses/ cache regardless of the `dataset` key used
# for other feature types) patches -- confirmed necessary via a 1-epoch
# dry run that failed with KeyError: 'set01' under the unpatched
# get_pose. This only imports PIE-side code, so there's no
# monkeypatch-ordering risk from a second dataset's import overwriting
# these (the bug that affected several cross-dataset scripts earlier
# this session doesn't apply to a single-dataset script).
import train_full_pie_nospeed as _pie_mod
_sfgru_mod.SFGRUTorch.get_pose = _pie_mod._safe_get_pose
_sfgru_mod.get_path = _pie_mod._patched_get_path
import utils as _u
_u.get_path = _pie_mod._patched_get_path
get_path = _pie_mod._patched_get_path

DATA_OPTS = {
    'fstride': 1,
    'subset': 'default',
    'data_split_type': 'default',
    'seq_type': 'crossing',
    'min_track_size': 75,
}
MODEL_OPTS = {
    'obs_input_type': ['local_box', 'local_context', 'pose', 'box'],
    'enlarge_ratio': 1.5,
    'pred_target_type': ['crossing'],
    'obs_length': 15,
    'time_to_event': 60,
    'dataset': 'pie_finetuned_context',
    'normalize_boxes': True,
}


def _batched_eval(model, inputs_cpu, labels, batch_size, device):
    """Runs model(...) in mini-batches, moving each batch to GPU only for
    the duration of its own forward pass -- keeps peak memory bounded
    regardless of how large inputs_cpu is."""
    model.eval()
    n = len(labels)
    preds = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch = [x[start:start + batch_size].to(device) for x in inputs_cpu]
            preds.append(model(batch).squeeze(-1).cpu())
    return torch.cat(preds).numpy().reshape(-1, 1)


def run_one_seed(seed, pose_backend='rtmpose', epochs=60, batch_size=16, lr=0.000005,
                 focal_alpha=0.5):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.environ['PIE_POSE_BACKEND'] = pose_backend

    method = FineTunedContextSFGRUTorch(device=device)
    imdb = PIE(data_path=PIE_DATA_DIR)

    log.info('seed=%d: preparing PIE train/val/test (raw local_context crops)...', seed)
    beh_train = imdb.generate_data_trajectory_sequence('train', **DATA_OPTS)
    beh_val = imdb.generate_data_trajectory_sequence('val', **DATA_OPTS)
    beh_test = imdb.generate_data_trajectory_sequence('test', **DATA_OPTS)

    model_opts = dict(MODEL_OPTS)
    train_data, data_types, data_sizes = method.get_data({'train': beh_train}, model_opts)
    train_inputs_cpu = [torch.from_numpy(np.asarray(x)).float() for x in train_data['train'][0]]
    train_labels = torch.from_numpy(np.asarray(train_data['train'][1])).float()

    val_data, _, _ = method.get_data({'test': beh_val}, model_opts)  # 'test' key = unbalanced, matches base convention
    val_inputs_cpu = [torch.from_numpy(np.asarray(x)).float() for x in val_data['test'][0]]
    val_labels_np = np.asarray(val_data['test'][1]).reshape(-1)

    model = method.build_model(data_types, data_sizes)
    # Only block5 + adapter + GRUs + output head are trainable -- the
    # frozen VGG blocks 1-4 already have requires_grad=False set inside
    # FineTunedContextEncoder, so filtering here is a belt-and-suspenders
    # double-check, not the primary mechanism.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in model.parameters())
    log.info('seed=%d: %d/%d trainable params (%.1f%%)', seed, n_trainable, n_total,
             100.0 * n_trainable / n_total)

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
    criterion = FocalBCELoss(alpha=focal_alpha, gamma=1.0)

    model_folder_name = f'pie_finetuned_context-seed{seed}'
    model_path, model_dir = get_path(save_folder=model_folder_name, save_root_folder='data/models',
                                     dataset='pie_finetuned_context', file_name='model.pt')

    n = len(train_labels)
    best_val_auc = -float('inf')
    best_epoch = -1
    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': [], 'val_auc': []}

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        epoch_correct = 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            batch_inputs = [x[idx].to(device) for x in train_inputs_cpu]
            batch_labels = train_labels[idx].to(device)

            optimizer.zero_grad()
            preds = model(batch_inputs).squeeze(-1)
            loss = criterion(preds, batch_labels.squeeze(-1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(idx)
            epoch_correct += ((preds > 0.5).float() == batch_labels.squeeze(-1)).sum().item()

        epoch_loss /= n
        epoch_acc = epoch_correct / n
        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_acc)

        val_preds_np = _batched_eval(model, val_inputs_cpu, val_labels_np, batch_size, device)
        val_loss = criterion(torch.from_numpy(val_preds_np.ravel()),
                             torch.from_numpy(val_labels_np.astype(np.float32))).item()
        val_acc = ((val_preds_np.ravel() > 0.5).astype(int) == val_labels_np).mean()
        val_auc = roc_auc_score(val_labels_np, val_preds_np.ravel())
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        history['val_auc'].append(val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            torch.save({'model_state_dict': model.state_dict(),
                       'data_types': data_types, 'data_sizes': data_sizes}, model_path)
        scheduler.step(val_loss)

        log.info('seed=%d epoch=%d/%d loss=%.4f acc=%.4f val_loss=%.4f val_acc=%.4f val_auc=%.4f',
                 seed, epoch + 1, epochs, epoch_loss, epoch_acc, val_loss, val_acc, val_auc)

    log.info('seed=%d: best_val_auc=%.4f at epoch %d', seed, best_val_auc, best_epoch + 1)

    # Final test-set evaluation using the best checkpoint, threshold=0.5
    # (matching this project's find_best_threshold-on-val convention would
    # need val-set threshold search too -- kept simple/0.5 here since this
    # is an in-domain sanity check, not the headline cross-dataset number).
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_data, _, _ = method.get_data({'test': beh_test}, model_opts)
    test_inputs_cpu = [torch.from_numpy(np.asarray(x)).float() for x in test_data['test'][0]]
    test_labels_np = np.asarray(test_data['test'][1]).reshape(-1)
    test_preds_np = _batched_eval(model, test_inputs_cpu, test_labels_np, batch_size, device)

    predictions = (test_preds_np >= 0.5).astype(int).ravel()
    acc = accuracy_score(test_labels_np, predictions)
    f1 = f1_score(test_labels_np, predictions, zero_division=0)
    prec = precision_score(test_labels_np, predictions, zero_division=0)
    rec = recall_score(test_labels_np, predictions, zero_division=0)
    auc = roc_auc_score(test_labels_np, test_preds_np.ravel())
    log.info('seed=%d TEST: Acc=%.4f AUC=%.4f F1=%.4f', seed, acc, auc, f1)

    return {'seed': seed, 'acc': acc, 'auc': auc, 'f1': f1, 'prec': prec, 'rec': rec,
            'model_path': model_dir, 'best_val_auc': best_val_auc, 'best_threshold': 0.5}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--backend', default='rtmpose', choices=['rtmpose'])
    args = parser.parse_args()

    runs = [run_one_seed(seed, pose_backend=args.backend) for seed in range(args.seeds)]

    accs = np.array([r['acc'] for r in runs])
    aucs = np.array([r['auc'] for r in runs])
    f1s = np.array([r['f1'] for r in runs])
    summary = {'acc_mean': float(accs.mean()), 'acc_std': float(accs.std()),
              'auc_mean': float(aucs.mean()), 'auc_std': float(aucs.std()),
              'f1_mean': float(f1s.mean()), 'f1_std': float(f1s.std()), 'n': len(runs)}

    out = os.path.join(RESULTS_DIR, f'full_pie_finetuned_context_results_{args.backend}.pkl')
    with open(out, 'wb') as f:
        pickle.dump({'runs': runs, 'summary': summary}, f)
    log.info('Saved: %s', out)

    print('\n' + '=' * 62)
    print(f'{"Backend":<12}{"Acc":>8}{"+/-":>8}{"AUC":>8}{"+/-":>8}{"F1":>8}{"+/-":>8}')
    print('-' * 62)
    print(f'{args.backend:<12}{summary["acc_mean"]:>8.4f}{summary["acc_std"]:>8.4f}'
          f'{summary["auc_mean"]:>8.4f}{summary["auc_std"]:>8.4f}'
          f'{summary["f1_mean"]:>8.4f}{summary["f1_std"]:>8.4f}')
    print('=' * 62)


if __name__ == '__main__':
    main()
