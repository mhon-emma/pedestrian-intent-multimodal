"""
train_full_combo.py
======================
Combines all three independently-validated fixes: domain-adversarial
training (GRL) + contrastive scale-invariance (box embedding) +
fine-tuned local_context (VGG16 block5 unfrozen + adapter instead of a
frozen never-adapted feature extractor). See sf_gru_torch_full_combo.py
for the full per-mechanism results and rationale.

Same UDA protocol as train_domain_adversarial_contrastive.py (source
labeled, target unlabeled train-split inputs only for the domain
adversary), same three-term loss (crossing + domain_loss_weight*domain
+ contrastive_weight*NT-Xent), model swapped for FullComboStackedGRU,
data prep swapped for FullComboSFGRUTorch[JAAD] (sf_gru_torch_full_combo_data.py
-- raw local_context images + box-jittering, both active via multiple
inheritance, verified to resolve correctly through the MRO).

Batched val/test evaluation (not the base SFGRUTorch.train/test's
whole-set-on-GPU pattern) -- same memory-safety reasoning as
train_full_{pie,jaad}_finetuned_context.py, since local_context is now
raw 224x224x3 images, not precomputed feature vectors. This departs
from train_domain_adversarial_contrastive.py's val-on-GPU-at-once
pattern (acceptable there since local_context was still precomputed
features in that script), so the val-loop structure here is adapted
from the finetuned_context scripts' batched pattern, not copied as-is.

JAAD-side overfitting caveat from the standalone fine-tuned-context
result (in-domain AUC 0.493+/-0.021 at chance, JAAD->PIE cross-dataset
0.354+/-0.193 with high seed variance, likely from 9.5M trainable
context-encoder params against only 195 JAAD training samples) is a
real risk here too -- FullComboStackedGRU exposes context_dropout for
this reason; --context_dropout defaults to 0.0 (off) so PIE-side
behavior is unaffected unless explicitly set for a JAAD-side run.

Protocol -- this is UDA, not joint supervised training
--------------------------------------------------------
Matches the train-on-SOURCE-test-on-TARGET protocol used everywhere else
in the paper (cross_dataset_eval.py etc.), so results are directly
comparable:
  - SOURCE dataset: full supervision (crossing-intent labels + domain
    label 0), train+val splits.
  - TARGET dataset: UNLABELED. Only its TRAIN split's inputs are used
    (domain label 1, crossing labels never touched); its VAL and TEST
    splits are never seen during training, exactly as in every other
    cross-dataset script in this repo. The domain classifier, reached
    through a Gradient Reversal Layer, is trained to distinguish source
    vs. target inputs; the feature extractor is trained (via the
    reversed gradient) to fool it, pushing the fused representation
    toward domain-invariance BEFORE ever touching the target's labels
    or test data.
  - Final evaluation: the adapted model's crossing-intent head is tested
    on the TARGET dataset's held-out TEST split -- the same
    train-on-one-test-on-other measurement as cross_dataset_eval.py,
    just with the target's unlabeled train inputs available at train
    time for the domain adversary (standard UDA setup; still never uses
    target labels).

Run once per direction (source->target), matching cross_dataset_eval.py's
pie_to_jaad / jaad_to_pie rows.

No speed (PIE-only signal, no JAAD equivalent). Modalities: local_box,
local_context, pose, box.

Usage
-----
  python train_domain_adversarial.py --source pie --target jaad --seeds 3
  python train_domain_adversarial.py --source jaad --target pie --seeds 3

Output
------
  results/domain_adversarial_<source>_to_<target>_rtmpose.pkl
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
from sf_gru_torch import SFGRUTorch, FocalBCELoss
from sf_gru_torch_contrastive import _nt_xent_loss
from sf_gru_torch_full_combo import FullComboStackedGRU
from sf_gru_torch_full_combo_data import FullComboSFGRUTorch, FullComboSFGRUTorchJAAD
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

# Reuse the exact monkeypatches (get_pose per dataset, JAAD's image loader,
# get_path, DATA_OPTS) from the existing nospeed scripts.
import train_full_pie_nospeed as _pie_mod
import train_full_jaad_nospeed_behonly as _jaad_mod

MODEL_OPTS_BASE = {
    'obs_input_type': ['local_box', 'local_context', 'pose', 'box'],  # no speed
    'enlarge_ratio': 1.5,
    'pred_target_type': ['crossing'],
    'obs_length': 15,
    'time_to_event': 60,
    'normalize_boxes': True,
}

DATASETS = {
    'pie':  {'getter_kind': 'pie'},
    'jaad': {'getter_kind': 'jaad'},
}


def apply_dataset_monkeypatches(dataset):
    """train_full_pie_nospeed.py and train_full_jaad_nospeed.py each patch
    sf_gru_torch.get_pose / get_path / load_images_crop_and_process at
    IMPORT time, as a module-level side effect -- fine when only one of
    them is ever imported in a process, but this script imports both (it
    needs both datasets), so whichever was imported LAST silently wins
    for every subsequent call, including the other dataset's pose/feature
    lookups. This must be called before EVERY get_raw_data/get_data call,
    not just once, since prep_features() -> get_data() -> get_pose() runs
    well after get_raw_data() returns and needs the correct patch active
    at call time, not import time."""
    if dataset == 'pie':
        _sfgru_mod.SFGRUTorch.get_pose = _pie_mod._safe_get_pose
        _sfgru_mod.get_path = _pie_mod._patched_get_path
        _sfgru_mod.SFGRUTorch.load_images_crop_and_process = SFGRUTorch.load_images_crop_and_process
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


def prep_features(method, beh_data, dataset_label, balanced=False):
    """balanced=True routes through get_data_sequence_balance (flip-
    augmentation to 50/50 class balance), matching every other training
    script in this paper for TRAIN/VAL splits of the labeled source
    dataset. balanced=False (the 'test' key) uses the natural, unbalanced
    distribution -- correct for: (a) any TEST split (paper-wide
    convention), and (b) the target dataset's train-split inputs here,
    since UDA's domain classifier doesn't need class-balanced crossing
    labels it never uses anyway."""
    model_opts = dict(MODEL_OPTS_BASE)
    model_opts['dataset'] = dataset_label
    key = 'train' if balanced else 'test'
    data, data_types, data_sizes = method.get_data({key: beh_data}, model_opts)
    inputs, labels = data[key]
    return inputs, np.asarray(labels), data_types, data_sizes


def prep_features_with_jitter(contrastive_method, beh_data, dataset_label):
    """Source TRAIN split only: contrastive_method must be a
    FullComboSFGRUTorch[JAAD] instance (not plain SFGRUTorch), whose
    get_data resolves to ContrastiveScaleInvariantSFGRU's override
    (via multiple inheritance, see sf_gru_torch_full_combo_data.py) so
    it also stashes a scale-jittered box view alongside generating raw
    local_context crops. Always balanced=True (train split, matching
    every other training script's source-domain convention) -- this
    path doesn't support an unbalanced call here since only the source
    train split ever needs the jittered view (contrastive loss is
    source-only, target/UDA data never touches it)."""
    model_opts = dict(MODEL_OPTS_BASE)
    model_opts['dataset'] = dataset_label
    data, data_types, data_sizes = contrastive_method.get_data({'train': beh_data}, model_opts)
    inputs, labels = data['train']
    box_jittered = contrastive_method._last_box_jittered['train']
    return inputs, np.asarray(labels), data_types, data_sizes, box_jittered


def lambda_schedule(progress, gamma=10.0):
    """DANN lambda ramp (Ganin & Lempitsky 2015, Eq. 6)."""
    return 2.0 / (1.0 + np.exp(-gamma * progress)) - 1.0


def _method_for(dataset):
    """Returns the correct FullComboSFGRUTorch[JAAD] instance for the
    given dataset's own path structure (local_context raw-crop path
    parsing differs: PIE has a set_id level, JAAD doesn't)."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return FullComboSFGRUTorchJAAD(device=device) if dataset == 'jaad' else FullComboSFGRUTorch(device=device)


def _batched_val_eval(model, inputs_cpu, labels_np, batch_size, device):
    model.eval()
    n = len(labels_np)
    preds = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch = [x[start:start + batch_size].to(device) for x in inputs_cpu]
            preds.append(model(batch).squeeze(-1).cpu())
    return torch.cat(preds).numpy().reshape(-1, 1)


def run_one_seed(seed, source, target, pose_backend, epochs=60, batch_size=16,
                 lr=3e-5, domain_loss_weight=1.0, gamma=10.0,
                 contrastive_weight=0.5, contrastive_temp=0.2, context_dropout=0.0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Both method instances must match each dataset's OWN path structure
    # -- source and target are usually different datasets here (UDA:
    # train on one, adapt to the other), so this is two distinct
    # objects, not one shared instance like the non-raw-image scripts
    # could get away with.
    src_method = _method_for(source)
    tgt_method = _method_for(target)

    log.info('seed=%d: preparing %s (source, labeled) train/val features (+ box jitter, raw context crops)...',
             seed, source)
    src_train_raw = get_raw_data(source, 'train', pose_backend)
    src_val_raw   = get_raw_data(source, 'val', pose_backend)
    # train: balanced (flip-augmented to 50/50) AND scale-jittered (for
    # the contrastive term). val: natural/unbalanced distribution
    # (matching every other script's convention) -- reuses src_method
    # (same class, so it still gets raw local_context crops) but via
    # prep_features's unjittered path since val never touches the
    # contrastive loss.
    src_train_in, src_train_lab, data_types, data_sizes, src_train_box_jittered = \
        prep_features_with_jitter(src_method, src_train_raw, f'{source}_combo_src')
    src_val_in, src_val_lab, _, _ = prep_features(
        src_method, src_val_raw, f'{source}_combo_src', balanced=False)

    box_idx = data_types.index('box')

    log.info('seed=%d: preparing %s (target, UNLABELED -- train split inputs only) features...', seed, target)
    tgt_train_raw = get_raw_data(target, 'train', pose_backend)
    # balanced=False: target labels are never used, so class balance is
    # irrelevant here; natural distribution is simplest and correct.
    tgt_train_in, tgt_train_lab_UNUSED, _, _ = prep_features(
        tgt_method, tgt_train_raw, f'{target}_combo_tgt', balanced=False)
    # tgt_train_lab_UNUSED is intentionally never used below -- UDA: target
    # labels are not available to the training procedure, only its inputs
    # (for the domain classifier) are used.

    n_src = len(src_train_lab)
    n_tgt = len(tgt_train_in[0])
    log.info('seed=%d: source(%s) train n=%d, target(%s) UNLABELED train n=%d',
             seed, source, n_src, target, n_tgt)

    model = FullComboStackedGRU(data_types, data_sizes, hidden_units=256,
                                context_dropout=context_dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)

    crossing_criterion = FocalBCELoss(alpha=0.5, gamma=1.0)
    domain_criterion = torch.nn.BCEWithLogitsLoss()

    # Raw local_context images make these CPU-resident tensors much
    # larger than every other UDA script in this project -- kept on CPU
    # (as before) and moved to GPU per-batch, same as
    # train_full_{pie,jaad}_finetuned_context.py's val/test loops, but
    # applied here to train/val alike since even the TRAIN tensor is now
    # sizable (though still moved to GPU per-batch during the loop below,
    # same as it always was -- only VAL changes from whole-set-on-GPU to
    # batched).
    src_inputs_t = [torch.from_numpy(np.asarray(x)).float() for x in src_train_in]
    src_labels_t = torch.from_numpy(src_train_lab).float()
    src_box_jittered_t = torch.from_numpy(np.asarray(src_train_box_jittered)).float()
    tgt_inputs_t = [torch.from_numpy(np.asarray(x)).float() for x in tgt_train_in]

    src_val_inputs_cpu = [torch.from_numpy(np.asarray(x)).float() for x in src_val_in]
    src_val_labels_np = np.asarray(src_val_lab).reshape(-1)

    n_batches_per_epoch = max(1, n_src // batch_size)
    total_steps = epochs * n_batches_per_epoch

    model_folder_name = f'full_combo_{source}_to_{target}-dlw{domain_loss_weight:g}-gamma{gamma:g}-seed{seed}'
    model_path, model_dir = _sfgru_mod.get_path(
        save_folder=model_folder_name, save_root_folder='data/models',
        dataset='full_combo', file_name='model.pt')

    best_val_auc = -float('inf')
    best_epoch = -1
    global_step = 0
    history = {'crossing_loss': [], 'domain_loss': [], 'contrastive_loss': [], 'val_auc': [], 'domain_acc': []}

    for epoch in range(epochs):
        model.train()
        src_perm = torch.randperm(n_src)
        # Target pool is typically a different size than source; sample
        # with replacement each batch so every source batch has a
        # same-sized target batch to compare against for the domain loss.
        epoch_crossing_loss = 0.0
        epoch_domain_loss = 0.0
        epoch_contrastive_loss = 0.0
        epoch_domain_correct = 0
        epoch_domain_total = 0

        for start in range(0, n_src, batch_size):
            src_idx = src_perm[start:start + batch_size]
            bs = len(src_idx)
            tgt_idx = torch.randint(0, n_tgt, (bs,))

            src_batch = [x[src_idx].to(device) for x in src_inputs_t]
            src_batch_labels = src_labels_t[src_idx].to(device).squeeze(-1)
            src_batch_box_jittered = src_box_jittered_t[src_idx].to(device)
            tgt_batch = [x[tgt_idx].to(device) for x in tgt_inputs_t]

            progress = global_step / max(1, total_steps)
            lam = lambda_schedule(progress, gamma=gamma)
            model.grl.set_lambda(lam)

            optimizer.zero_grad()

            # Crossing-intent + domain loss: source only (labeled),
            # PLUS the source batch's own box embedding for the
            # contrastive term below.
            src_crossing_probs, src_domain_logits, src_box_embed = model(
                src_batch, return_domain_logits=True, return_box_embedding=True)
            crossing_loss = crossing_criterion(src_crossing_probs.squeeze(-1), src_batch_labels)

            # Contrastive term: source batch's box embedding vs. the SAME
            # batch's scale-jittered box view's embedding (positive
            # pairs), matching sf_gru_torch_contrastive.py's training
            # loop exactly -- jittered_inputs replaces only the box
            # modality's tensor, everything else (local_box,
            # local_context, pose) stays the real (unjittered) source
            # batch, so the jittered forward pass isn't a second
            # domain/crossing prediction, just a second box embedding.
            jittered_inputs = list(src_batch)
            jittered_inputs[box_idx] = src_batch_box_jittered
            _, _, src_box_embed_jittered = model(
                jittered_inputs, return_domain_logits=True, return_box_embedding=True)
            contrastive_loss = _nt_xent_loss(src_box_embed, src_box_embed_jittered, temperature=contrastive_temp)

            # Domain loss: source (label 0) + target (label 1), both
            # through the same GRL-gated domain classifier. Target inputs
            # never touch crossing_criterion or the contrastive term.
            _, tgt_domain_logits, _ = model(tgt_batch, return_domain_logits=True, return_box_embedding=True)
            domain_logits = torch.cat([src_domain_logits.squeeze(-1), tgt_domain_logits.squeeze(-1)], dim=0)
            domain_labels = torch.cat([
                torch.zeros(bs, device=device), torch.ones(bs, device=device)
            ], dim=0)
            domain_loss = domain_criterion(domain_logits, domain_labels)

            loss = (crossing_loss + domain_loss_weight * domain_loss
                    + contrastive_weight * contrastive_loss)
            loss.backward()
            optimizer.step()

            epoch_crossing_loss += crossing_loss.item() * bs
            epoch_domain_loss += domain_loss.item() * (2 * bs)
            epoch_contrastive_loss += contrastive_loss.item() * bs
            domain_preds = (torch.sigmoid(domain_logits) > 0.5).float()
            epoch_domain_correct += (domain_preds == domain_labels).sum().item()
            epoch_domain_total += 2 * bs
            global_step += 1

        epoch_crossing_loss /= n_src
        epoch_domain_loss /= epoch_domain_total
        epoch_contrastive_loss /= n_src
        domain_acc = epoch_domain_correct / epoch_domain_total
        history['crossing_loss'].append(epoch_crossing_loss)
        history['domain_loss'].append(epoch_domain_loss)
        history['contrastive_loss'].append(epoch_contrastive_loss)
        history['domain_acc'].append(domain_acc)

        val_preds_np = _batched_val_eval(model, src_val_inputs_cpu, src_val_labels_np, batch_size, device)
        val_auc = roc_auc_score(src_val_labels_np, val_preds_np.ravel())
        history['val_auc'].append(val_auc)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            torch.save({
                'model_state_dict': model.state_dict(),
                'data_types': data_types,
                'data_sizes': data_sizes,
            }, model_path)
        val_loss_for_sched = crossing_criterion(
            torch.from_numpy(val_preds_np.ravel()),
            torch.from_numpy(src_val_labels_np.astype(np.float32))).item()
        scheduler.step(val_loss_for_sched)

        log.info('seed=%d [%s->%s] epoch=%d/%d  crossing_loss=%.4f  domain_loss=%.4f  '
                 'contrastive_loss=%.4f  domain_acc=%.3f (lambda=%.3f; ->0.5 domain-invariant)  '
                 'src_val_auc=%.4f',
                 seed, source, target, epoch + 1, epochs, epoch_crossing_loss,
                 epoch_domain_loss, epoch_contrastive_loss, domain_acc, lam, val_auc)

    log.info('seed=%d [%s->%s]: best src_val_auc=%.4f at epoch %d. Final domain_acc=%.3f',
             seed, source, target, best_val_auc, best_epoch + 1, history['domain_acc'][-1])

    model_opts_path, _ = _sfgru_mod.get_path(
        save_folder=model_folder_name, save_root_folder='data/models',
        dataset='full_combo', file_name='model_opts.pkl')
    with open(model_opts_path, 'wb') as fid:
        pickle.dump(MODEL_OPTS_BASE, fid, pickle.HIGHEST_PROTOCOL)

    return {
        'seed': seed, 'source': source, 'target': target, 'model_path': model_dir,
        'best_val_auc': best_val_auc, 'best_epoch': best_epoch,
        'final_domain_acc': history['domain_acc'][-1], 'history': history,
        'data_types': data_types, 'data_sizes': data_sizes,
    }


def evaluate_on_target_test(run, target, pose_backend, batch_size=16):
    """The actual cross-dataset transfer measurement: adapted model's
    crossing-intent head evaluated on the TARGET's held-out TEST split --
    never touched during training (only target's train-split inputs were
    used, unlabeled, for the domain adversary). Batched (raw
    local_context images make whole-set-on-GPU risky, same reasoning as
    train_full_{pie,jaad}_finetuned_context.py)."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(os.path.join(run['model_path'], 'model.pt'), map_location=device)
    model = FullComboStackedGRU(run['data_types'], run['data_sizes'], hidden_units=256).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    method = _method_for(target)
    beh_test = get_raw_data(target, 'test', pose_backend)
    inputs, labels, _, _ = prep_features(method, beh_test, f'{target}_combo_tgt_test')
    inputs_cpu = [torch.from_numpy(np.asarray(x)).float() for x in inputs]
    labels_np = np.asarray(labels).reshape(-1)

    preds = []
    with torch.no_grad():
        for start in range(0, len(labels_np), batch_size):
            batch = [x[start:start + batch_size].to(device) for x in inputs_cpu]
            preds.append(model(batch).squeeze(-1).cpu())
    preds = torch.cat(preds).numpy().ravel()

    auc = roc_auc_score(labels_np, preds)
    acc = accuracy_score(labels_np, (preds >= 0.5).astype(int))
    f1 = f1_score(labels_np, (preds >= 0.5).astype(int), zero_division=0)
    log.info('  [target-test=%s] Acc=%.4f AUC=%.4f F1=%.4f', target, acc, auc, f1)
    return {'auc': auc, 'acc': acc, 'f1': f1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, choices=['pie', 'jaad'])
    parser.add_argument('--target', required=True, choices=['pie', 'jaad'])
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--backend', default='rtmpose', choices=['rtmpose'])
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--domain_loss_weight', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=10.0,
                        help='DANN lambda-ramp steepness (Ganin & Lempitsky Eq. 6). '
                             'Lower = slower ramp, gives the domain classifier more '
                             'epochs before adversarial pressure increases.')
    parser.add_argument('--context_dropout', type=float, default=0.0,
                        help='Dropout on the local_context adapter output, before it '
                             'enters the GRU stack. Default 0.0 (off) -- available for '
                             'JAAD-side runs given the standalone fine-tuned-context '
                             'experiment showed JAAD overfitting risk (9.5M trainable '
                             'context-encoder params against only 195 training samples).')
    args = parser.parse_args()
    if args.source == args.target:
        raise ValueError('--source and --target must differ')

    runs = []
    target_results = []
    for seed in range(args.seeds):
        run = run_one_seed(seed, args.source, args.target, args.backend,
                           epochs=args.epochs, domain_loss_weight=args.domain_loss_weight,
                           gamma=args.gamma, context_dropout=args.context_dropout)
        runs.append(run)
        log.info('=== seed=%d: evaluating on %s TEST split (true cross-dataset transfer) ===',
                 seed, args.target)
        result = evaluate_on_target_test(run, args.target, args.backend)
        target_results.append(result)

    aucs = [r['auc'] for r in target_results]
    accs = [r['acc'] for r in target_results]
    f1s = [r['f1'] for r in target_results]
    domain_accs = [r['final_domain_acc'] for r in runs]

    summary = {
        'auc_mean': float(np.mean(aucs)), 'auc_std': float(np.std(aucs)),
        'acc_mean': float(np.mean(accs)), 'acc_std': float(np.std(accs)),
        'f1_mean': float(np.mean(f1s)), 'f1_std': float(np.std(f1s)),
        'domain_acc_mean': float(np.mean(domain_accs)), 'domain_acc_std': float(np.std(domain_accs)),
        'n': len(runs), 'source': args.source, 'target': args.target,
        'domain_loss_weight': args.domain_loss_weight, 'gamma': args.gamma,
    }

    dlw_tag = f'dlw{args.domain_loss_weight:g}'
    gamma_tag = f'gamma{args.gamma:g}'
    cd_tag = f'_cd{args.context_dropout:g}' if args.context_dropout > 0 else ''
    out = os.path.join(RESULTS_DIR,
                       f'full_combo_{args.source}_to_{args.target}_{args.backend}_{dlw_tag}_{gamma_tag}{cd_tag}.pkl')
    with open(out, 'wb') as f:
        pickle.dump({'runs': runs, 'target_results': target_results, 'summary': summary}, f)
    log.info('Saved: %s', out)

    print('\n' + '=' * 78)
    print(f'Domain-adversarial SF-GRU (UDA): {args.source} (labeled) -> {args.target} (unlabeled, adapt)')
    print(f'Evaluated on {args.target}\'s held-out TEST split -- true cross-dataset transfer')
    print('-' * 78)
    print(f'{args.target} test AUC:  {summary["auc_mean"]:.4f} +/- {summary["auc_std"]:.4f}')
    print(f'{args.target} test Acc:  {summary["acc_mean"]:.4f} +/- {summary["acc_std"]:.4f}')
    print(f'{args.target} test F1:   {summary["f1_mean"]:.4f} +/- {summary["f1_std"]:.4f}')
    print(f'Final domain classifier accuracy: {summary["domain_acc_mean"]:.4f} +/- {summary["domain_acc_std"]:.4f}')
    print('  (0.5 = fully domain-invariant fused representation; 1.0 = domain trivially recoverable)')
    print('=' * 78)


if __name__ == '__main__':
    main()
