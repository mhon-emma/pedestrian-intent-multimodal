"""
train_domain_adversarial_contrastive_crossattn_context.py
==============================================================
Extends the context-targeted domain-adversarial idea (found to match
the full-representation headline recipe on SF-GRU's stacked fusion,
train_domain_adversarial_contrastive_context.py) to the cross-attention
architecture: GRL + domain classifier read local_context's own
independent encoder output (pre-attention) instead of the post-
attention fused representation
(sf_gru_torch_domain_adversarial_contrastive_crossattn_context.py).

Direct comparison point: cross-attention's full-representation result
(train_domain_adversarial_contrastive_crossattn.py, Table tab:crossattn:
PIE->JAAD 0.559+/-0.027, JAAD->PIE 0.446+/-0.063, both n=6, default
contrastive hyperparameters). Same protocol here: default contrastive
hyperparameters, n=6 seeds, both directions -- isolating the tap-point
change from any hyperparameter effect.

All checkpoint/output paths tagged '_crossattn_context', distinct from
both the full-representation crossattn scripts and the stacked-GRU
context-targeted scripts.

Usage
-----
  python train_domain_adversarial_contrastive_crossattn_context.py --source pie --target jaad --seeds 6
  python train_domain_adversarial_contrastive_crossattn_context.py --source jaad --target pie --seeds 6

Output
------
  results/domain_adversarial_contrastive_crossattn_context_<source>_to_<target>_rtmpose_dlw1_gamma10.pkl
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
from sf_gru_torch_contrastive import ContrastiveScaleInvariantSFGRU, _nt_xent_loss
from sf_gru_torch_domain_adversarial_contrastive_crossattn_context import (
    CrossAttentionDomainAdversarialContrastiveContextStackedGRU)
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import train_full_pie_nospeed as _pie_mod
import train_full_jaad_nospeed_behonly as _jaad_mod

MODEL_OPTS_BASE = {
    'obs_input_type': ['local_box', 'local_context', 'pose', 'box'],
    'enlarge_ratio': 1.5,
    'pred_target_type': ['crossing'],
    'obs_length': 15,
    'time_to_event': 60,
    'normalize_boxes': True,
}


def apply_dataset_monkeypatches(dataset):
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
    model_opts = dict(MODEL_OPTS_BASE)
    model_opts['dataset'] = dataset_label
    key = 'train' if balanced else 'test'
    data, data_types, data_sizes = method.get_data({key: beh_data}, model_opts)
    inputs, labels = data[key]
    return inputs, np.asarray(labels), data_types, data_sizes


def prep_features_with_jitter(contrastive_method, beh_data, dataset_label):
    model_opts = dict(MODEL_OPTS_BASE)
    model_opts['dataset'] = dataset_label
    data, data_types, data_sizes = contrastive_method.get_data({'train': beh_data}, model_opts)
    inputs, labels = data['train']
    box_jittered = contrastive_method._last_box_jittered['train']
    return inputs, np.asarray(labels), data_types, data_sizes, box_jittered


def lambda_schedule(progress, gamma=10.0):
    return 2.0 / (1.0 + np.exp(-gamma * progress)) - 1.0


def run_one_seed(seed, source, target, pose_backend, epochs=60, batch_size=32,
                 lr=3e-5, domain_loss_weight=1.0, gamma=10.0,
                 contrastive_weight=0.5, contrastive_temp=0.2):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    method = SFGRUTorch(device=device)
    contrastive_method = ContrastiveScaleInvariantSFGRU(device=device)

    log.info('seed=%d: preparing %s (source, labeled) train/val features (+ box jitter)...', seed, source)
    src_train_raw = get_raw_data(source, 'train', pose_backend)
    src_val_raw   = get_raw_data(source, 'val', pose_backend)
    src_train_in, src_train_lab, data_types, data_sizes, src_train_box_jittered = \
        prep_features_with_jitter(contrastive_method, src_train_raw, f'{source}_da_crossattn_ctx_src')
    src_val_in, src_val_lab, _, _ = prep_features(
        method, src_val_raw, f'{source}_da_crossattn_ctx_src', balanced=False)

    box_idx = data_types.index('box')

    log.info('seed=%d: preparing %s (target, UNLABELED -- train split inputs only) features...', seed, target)
    tgt_train_raw = get_raw_data(target, 'train', pose_backend)
    tgt_train_in, tgt_train_lab_UNUSED, _, _ = prep_features(
        method, tgt_train_raw, f'{target}_da_crossattn_ctx_tgt', balanced=False)

    n_src = len(src_train_lab)
    n_tgt = len(tgt_train_in[0])
    log.info('seed=%d: source(%s) train n=%d, target(%s) UNLABELED train n=%d',
             seed, source, n_src, target, n_tgt)

    model = CrossAttentionDomainAdversarialContrastiveContextStackedGRU(
        data_types, data_sizes, hidden_units=256, anchor='box').to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)

    crossing_criterion = FocalBCELoss(alpha=0.5, gamma=1.0)
    domain_criterion = torch.nn.BCEWithLogitsLoss()

    src_inputs_t = [torch.from_numpy(np.asarray(x)).float() for x in src_train_in]
    src_labels_t = torch.from_numpy(src_train_lab).float()
    src_box_jittered_t = torch.from_numpy(np.asarray(src_train_box_jittered)).float()
    tgt_inputs_t = [torch.from_numpy(np.asarray(x)).float() for x in tgt_train_in]

    src_val_inputs_t = [torch.from_numpy(np.asarray(x)).float().to(device) for x in src_val_in]
    src_val_labels_t = torch.from_numpy(src_val_lab).float().to(device)

    n_batches_per_epoch = max(1, n_src // batch_size)
    total_steps = epochs * n_batches_per_epoch

    cw_tag = f'-cw{contrastive_weight:g}' if contrastive_weight != 0.5 else ''
    ct_tag = f'-ct{contrastive_temp:g}' if contrastive_temp != 0.2 else ''
    model_folder_name = (f'domain_adversarial_contrastive_crossattn_context_{source}_to_{target}-'
                         f'dlw{domain_loss_weight:g}-gamma{gamma:g}{cw_tag}{ct_tag}-seed{seed}')
    model_path, model_dir = _sfgru_mod.get_path(
        save_folder=model_folder_name, save_root_folder='data/models',
        dataset='domain_adversarial_contrastive_crossattn_context', file_name='model.pt')

    best_val_auc = -float('inf')
    best_epoch = -1
    global_step = 0
    history = {'crossing_loss': [], 'domain_loss': [], 'contrastive_loss': [], 'val_auc': [], 'domain_acc': []}

    for epoch in range(epochs):
        model.train()
        src_perm = torch.randperm(n_src)
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

            src_crossing_probs, src_domain_logits, src_box_embed = model(
                src_batch, return_domain_logits=True, return_box_embedding=True)
            crossing_loss = crossing_criterion(src_crossing_probs.squeeze(-1), src_batch_labels)

            jittered_inputs = list(src_batch)
            jittered_inputs[box_idx] = src_batch_box_jittered
            _, _, src_box_embed_jittered = model(
                jittered_inputs, return_domain_logits=True, return_box_embedding=True)
            contrastive_loss = _nt_xent_loss(src_box_embed, src_box_embed_jittered, temperature=contrastive_temp)

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

        model.eval()
        with torch.no_grad():
            val_preds = model(src_val_inputs_t).squeeze(-1)
            val_auc = roc_auc_score(src_val_labels_t.cpu().numpy().ravel(), val_preds.cpu().numpy())
            history['val_auc'].append(val_auc)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch = epoch
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'data_types': data_types,
                    'data_sizes': data_sizes,
                }, model_path)
        val_loss_for_sched = crossing_criterion(val_preds, src_val_labels_t.squeeze(-1)).item()
        scheduler.step(val_loss_for_sched)

        log.info('seed=%d [%s->%s crossattn_context] epoch=%d/%d  crossing_loss=%.4f  domain_loss=%.4f  '
                 'contrastive_loss=%.4f  domain_acc=%.3f (lambda=%.3f; ->0.5 domain-invariant)  '
                 'src_val_auc=%.4f',
                 seed, source, target, epoch + 1, epochs, epoch_crossing_loss,
                 epoch_domain_loss, epoch_contrastive_loss, domain_acc, lam, val_auc)

    log.info('seed=%d [%s->%s crossattn_context]: best src_val_auc=%.4f at epoch %d. Final domain_acc=%.3f',
             seed, source, target, best_val_auc, best_epoch + 1, history['domain_acc'][-1])

    model_opts_path, _ = _sfgru_mod.get_path(
        save_folder=model_folder_name, save_root_folder='data/models',
        dataset='domain_adversarial_contrastive_crossattn_context', file_name='model_opts.pkl')
    with open(model_opts_path, 'wb') as fid:
        pickle.dump(MODEL_OPTS_BASE, fid, pickle.HIGHEST_PROTOCOL)

    return {
        'seed': seed, 'source': source, 'target': target, 'model_path': model_dir,
        'best_val_auc': best_val_auc, 'best_epoch': best_epoch,
        'final_domain_acc': history['domain_acc'][-1], 'history': history,
        'data_types': data_types, 'data_sizes': data_sizes,
    }


def evaluate_on_target_test(run, target, pose_backend):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(os.path.join(run['model_path'], 'model.pt'), map_location=device)
    model = CrossAttentionDomainAdversarialContrastiveContextStackedGRU(
        run['data_types'], run['data_sizes'], hidden_units=256, anchor='box').to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    method = SFGRUTorch(device=device)
    beh_test = get_raw_data(target, 'test', pose_backend)
    inputs, labels, _, _ = prep_features(method, beh_test, f'{target}_da_crossattn_ctx_tgt_test')
    inputs_t = [torch.from_numpy(np.asarray(x)).float().to(device) for x in inputs]
    with torch.no_grad():
        preds = model(inputs_t).squeeze(-1).cpu().numpy()
    auc = roc_auc_score(labels.ravel(), preds)
    acc = accuracy_score(labels.ravel(), (preds >= 0.5).astype(int))
    f1 = f1_score(labels.ravel(), (preds >= 0.5).astype(int), zero_division=0)
    log.info('  [target-test=%s crossattn_context] Acc=%.4f AUC=%.4f F1=%.4f', target, acc, auc, f1)
    return {'auc': auc, 'acc': acc, 'f1': f1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, choices=['pie', 'jaad'])
    parser.add_argument('--target', required=True, choices=['pie', 'jaad'])
    parser.add_argument('--seeds', type=int, default=6)
    parser.add_argument('--backend', default='rtmpose', choices=['rtmpose'])
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--domain_loss_weight', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=10.0)
    parser.add_argument('--contrastive_weight', type=float, default=0.5)
    parser.add_argument('--contrastive_temp', type=float, default=0.2)
    args = parser.parse_args()
    if args.source == args.target:
        raise ValueError('--source and --target must differ')

    runs = []
    target_results = []
    for seed in range(args.seeds):
        run = run_one_seed(seed, args.source, args.target, args.backend,
                           epochs=args.epochs, domain_loss_weight=args.domain_loss_weight,
                           gamma=args.gamma, contrastive_weight=args.contrastive_weight,
                           contrastive_temp=args.contrastive_temp)
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
        'architecture': 'crossattn_context',
    }

    dlw_tag = f'dlw{args.domain_loss_weight:g}'
    gamma_tag = f'gamma{args.gamma:g}'
    cw_tag = f'_cw{args.contrastive_weight:g}' if args.contrastive_weight != 0.5 else ''
    ct_tag = f'_ct{args.contrastive_temp:g}' if args.contrastive_temp != 0.2 else ''
    out = os.path.join(RESULTS_DIR,
                       f'domain_adversarial_contrastive_crossattn_context_{args.source}_to_{args.target}_{args.backend}_{dlw_tag}_{gamma_tag}{cw_tag}{ct_tag}.pkl')
    with open(out, 'wb') as f:
        pickle.dump({'runs': runs, 'target_results': target_results, 'summary': summary}, f)
    log.info('Saved: %s', out)

    print('\n' + '=' * 78)
    print(f'CrossAttention (context-targeted) Domain-adversarial SF-GRU (UDA): {args.source} (labeled) -> {args.target} (unlabeled, adapt)')
    print(f'Evaluated on {args.target}\'s held-out TEST split -- true cross-dataset transfer')
    print('-' * 78)
    print(f'{args.target} test AUC:  {summary["auc_mean"]:.4f} +/- {summary["auc_std"]:.4f}')
    print(f'{args.target} test Acc:  {summary["acc_mean"]:.4f} +/- {summary["acc_std"]:.4f}')
    print(f'{args.target} test F1:   {summary["f1_mean"]:.4f} +/- {summary["f1_std"]:.4f}')
    print(f'Final domain classifier accuracy: {summary["domain_acc_mean"]:.4f} +/- {summary["domain_acc_std"]:.4f}')
    print('=' * 78)


if __name__ == '__main__':
    main()
