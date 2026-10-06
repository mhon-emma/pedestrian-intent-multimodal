"""
dinov2_localbox_probe.py
============================
Companion/contrast case to dinov2_context_probe.py. That script tested
whether local_context's diagnosed leakage survives a different frozen
backbone (DINOv2 in place of VGG16) -- it does (separability does not
drop). This script applies the SAME swap to local_box instead, as a
contrast case: local_box was NOT implicated by the modality-zeroing
ablation (zeroing it does not help transfer the way zeroing
local_context does), so this tests a different prediction. If DINOv2
inflates separability for ANY visual modality regardless of content
(e.g. a generic artifact of self-supervised ViT features carrying more
easily-decodable low-level statistics than supervised CNN features),
we would expect local_box's separability to rise similarly under
DINOv2. If the effect is specific to local_context's content (the
surrounding, pedestrian-excluded scene crop) rather than to DINOv2 as
an encoder, local_box's separability should stay roughly where it was
with VGG16 regardless of backbone.

Method: identical structure to dinov2_context_probe.py, with the
VGG16/DINOv2 assignment swapped: local_box now uses
DinoV2ContextSFGRUTorch (the class is backbone-generic despite its
name -- it swaps whichever crop_type's forward pass reaches
_vgg_forward on that instance, see sf_gru_torch_dinov2_context.py),
local_context/pose/box stay on plain VGG16/unchanged SFGRUTorch.

Usage
-----
  python dinov2_localbox_probe.py

Output
------
  results/dinov2_localbox_probe.pkl
"""

import logging
import os
import pickle
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

SFGRU_DIR     = '/usr1/home/mehon/emma_pedestrian-intent-multimodal'
PIE_UTIL_DIR  = '/usr1/home/mehon/PIE/utilities'
PIE_DATA_DIR  = '/usr1/home/mehon/data_root/pie'
JAAD_UTIL_DIR = '/usr1/home/mehon/JAAD'
JAAD_DATA_DIR = '/usr1/home/mehon/JAAD'
RESULTS_DIR   = os.path.join(SFGRU_DIR, 'results')

sys.path.insert(0, SFGRU_DIR)
sys.path.append(PIE_UTIL_DIR)
sys.path.append(JAAD_UTIL_DIR)
os.chdir(SFGRU_DIR)

import sf_gru_torch as _sfgru_mod
from pie_data import PIE
from jaad_data import JAAD
from sf_gru_torch import SFGRUTorch, FocalBCELoss
from sf_gru_torch_dinov2_context import DinoV2ContextSFGRUTorch

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


def get_raw_data(dataset, split, pose_backend='rtmpose'):
    apply_dataset_monkeypatches(dataset)
    if dataset == 'pie':
        imdb = PIE(data_path=PIE_DATA_DIR)
        os.environ['PIE_POSE_BACKEND'] = pose_backend
        return imdb.generate_data_trajectory_sequence(split, **_pie_mod.DATA_OPTS)
    else:
        imdb = JAAD(data_path=JAAD_DATA_DIR)
        os.environ['JAAD_POSE_BACKEND'] = pose_backend
        return imdb.generate_data_trajectory_sequence(split, **_jaad_mod.DATA_OPTS)


def get_merged_features(beh_data, dataset, dataset_key, balanced, device):
    """Mirror of dinov2_context_probe.py's function with the backbone
    assignment swapped: local_context+pose+box via plain VGG16/unchanged
    SFGRUTorch (reusing the existing cache under `dataset_key`),
    local_box via DinoV2ContextSFGRUTorch under a SEPARATE cache
    namespace (`dataset_key + '_dinov2box'`, distinct from
    dinov2_context_probe.py's '_dinov2ctx' namespace -- never collides)."""
    apply_dataset_monkeypatches(dataset)
    vgg_method = SFGRUTorch(device=device)
    dino_method = DinoV2ContextSFGRUTorch(device=device)

    key = 'train' if balanced else 'test'

    model_opts_vgg = dict(MODEL_OPTS_BASE)
    model_opts_vgg['obs_input_type'] = ['local_context', 'pose', 'box']
    model_opts_vgg['dataset'] = dataset_key
    data_vgg, types_vgg, sizes_vgg = vgg_method.get_data({key: beh_data}, model_opts_vgg)
    inputs_vgg, labels = data_vgg[key]

    # Re-apply the dataset-specific monkeypatch immediately before the
    # DINOv2 call -- see dinov2_context_probe.py's get_merged_features
    # for why this is needed defensively between two get_data calls in
    # one function (empirically observed patch leakage between the two
    # passes, not just across function calls).
    apply_dataset_monkeypatches(dataset)

    model_opts_dino = dict(MODEL_OPTS_BASE)
    model_opts_dino['obs_input_type'] = ['local_box']
    model_opts_dino['dataset'] = f'{dataset_key}_dinov2box'
    data_dino, types_dino, sizes_dino = dino_method.get_data({key: beh_data}, model_opts_dino)
    inputs_dino, labels_dino = data_dino[key]
    assert np.array_equal(np.asarray(labels_dino), np.asarray(labels)), \
        'label mismatch between VGG and DINOv2 extraction passes'

    merged_types = ['local_box', 'local_context', 'pose', 'box']
    merged_inputs = [
        inputs_dino[types_dino.index('local_box')],
        inputs_vgg[types_vgg.index('local_context')],
        inputs_vgg[types_vgg.index('pose')],
        inputs_vgg[types_vgg.index('box')],
    ]
    merged_sizes = [
        sizes_dino[types_dino.index('local_box')],
        sizes_vgg[types_vgg.index('local_context')],
        sizes_vgg[types_vgg.index('pose')],
        sizes_vgg[types_vgg.index('box')],
    ]
    return merged_inputs, np.asarray(labels), merged_types, merged_sizes


def train_baseline(source_dataset, device, epochs=40, batch_size=32, lr=3e-5):
    from sf_gru_torch import StackedGRU
    from sklearn.metrics import roc_auc_score as _auc

    train_raw = get_raw_data(source_dataset, 'train')
    val_raw = get_raw_data(source_dataset, 'val')

    train_in, train_lab, data_types, data_sizes = get_merged_features(
        train_raw, source_dataset, f'{source_dataset}_dinov2box_src', balanced=True, device=device)
    val_in, val_lab, _, _ = get_merged_features(
        val_raw, source_dataset, f'{source_dataset}_dinov2box_src', balanced=False, device=device)

    model = StackedGRU(data_types, data_sizes, hidden_units=256, weight_decay=1e-4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = FocalBCELoss(alpha=0.5, gamma=1.0)

    train_inputs_t = [torch.from_numpy(np.asarray(x)).float().to(device) for x in train_in]
    train_labels_t = torch.from_numpy(train_lab).float().to(device).squeeze(-1)
    val_inputs_t = [torch.from_numpy(np.asarray(x)).float().to(device) for x in val_in]
    val_labels_t = torch.from_numpy(val_lab).float().to(device)

    n = len(train_lab)
    best_auc = -1.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            batch = [x[idx] for x in train_inputs_t]
            labels_b = train_labels_t[idx]
            optimizer.zero_grad()
            preds = model(batch).squeeze(-1)
            loss = criterion(preds, labels_b)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(val_inputs_t).squeeze(-1).cpu().numpy()
            val_auc = _auc(val_labels_t.cpu().numpy().ravel(), val_preds)
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        log.info('[%s dinov2box baseline] epoch=%d/%d val_auc=%.4f (best=%.4f)',
                 source_dataset, epoch + 1, epochs, val_auc, best_auc)

    model.load_state_dict(best_state)
    model.eval()
    return model, data_types, data_sizes, best_auc


@torch.no_grad()
def extract_representation(model, inputs, data_types, tap='fused', device='cuda'):
    inputs_t = [torch.from_numpy(np.asarray(x)).float().to(device) for x in inputs]
    x = None
    box_hidden = None
    box_idx = data_types.index('local_box')
    for i, gru in enumerate(model.grus):
        is_last = (i == len(model.grus) - 1)
        seq_in = inputs_t[0] if i == 0 else torch.cat([x, inputs_t[i]], dim=2)
        out, h = gru(seq_in)
        if i == box_idx:
            box_hidden = h.squeeze(0)
        x = out if not is_last else h.squeeze(0)
    if tap == 'fused':
        return x.cpu().numpy()
    return box_hidden.cpu().numpy()


def run_probe(pie_repr, jaad_repr, label='probe', n_repeats=5, seed_base=0):
    X = np.concatenate([pie_repr, jaad_repr], axis=0)
    y = np.concatenate([np.zeros(len(pie_repr)), np.ones(len(jaad_repr))])
    linear_accs, mlp_accs = [], []
    for r in range(n_repeats):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=seed_base + r, stratify=y)
        lin = LogisticRegression(max_iter=2000, C=1.0)
        lin.fit(X_train, y_train)
        linear_accs.append(accuracy_score(y_test, lin.predict(X_test)))
        mlp = MLPClassifier(hidden_layer_sizes=(32,), max_iter=2000,
                            early_stopping=True, random_state=seed_base + r)
        mlp.fit(X_train, y_train)
        mlp_accs.append(accuracy_score(y_test, mlp.predict(X_test)))
    linear_accs, mlp_accs = np.array(linear_accs), np.array(mlp_accs)
    log.info('%s: linear=%.4f+/-%.4f  mlp=%.4f+/-%.4f (n_pie=%d, n_jaad=%d)',
             label, linear_accs.mean(), linear_accs.std(), mlp_accs.mean(), mlp_accs.std(),
             len(pie_repr), len(jaad_repr))
    return {'acc_mean': float(linear_accs.mean()), 'acc_std': float(linear_accs.std()),
            'mlp_acc_mean': float(mlp_accs.mean()), 'mlp_acc_std': float(mlp_accs.std()),
            'n_repeats': n_repeats, 'n_pie': len(pie_repr), 'n_jaad': len(jaad_repr)}


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}

    log.info('=== Training PIE baseline (DINOv2 local_box) ===')
    pie_model, pie_types, pie_sizes, pie_val_auc = train_baseline('pie', device)
    results['pie_indomain_val_auc'] = pie_val_auc

    log.info('=== Extracting PIE test features (DINOv2 local_box) ===')
    pie_test_raw = get_raw_data('pie', 'test')
    pie_test_in, pie_test_lab, pie_test_types, _ = get_merged_features(
        pie_test_raw, 'pie', 'pie_dinov2box_tgt_test', balanced=False, device=device)
    with torch.no_grad():
        inputs_t = [torch.from_numpy(np.asarray(x)).float().to(device) for x in pie_test_in]
        preds = pie_model(inputs_t).squeeze(-1).cpu().numpy()
    pie_test_auc_indomain = roc_auc_score(pie_test_lab.ravel(), preds)
    results['pie_indomain_test_auc'] = float(pie_test_auc_indomain)
    log.info('PIE in-domain TEST AUC (DINOv2 local_box): %.4f', pie_test_auc_indomain)

    pie_repr_fused = extract_representation(pie_model, pie_test_in, pie_test_types, tap='fused', device=device)
    pie_repr_box = extract_representation(pie_model, pie_test_in, pie_test_types, tap='box', device=device)

    log.info('=== Extracting JAAD test features (DINOv2 local_box) ===')
    jaad_test_raw = get_raw_data('jaad', 'test')
    jaad_test_in, jaad_test_lab, jaad_test_types, _ = get_merged_features(
        jaad_test_raw, 'jaad', 'jaad_dinov2box_tgt_test', balanced=False, device=device)

    jaad_repr_fused = extract_representation(pie_model, jaad_test_in, jaad_test_types, tap='fused', device=device)
    jaad_repr_box = extract_representation(pie_model, jaad_test_in, jaad_test_types, tap='box', device=device)

    log.info('--- Probe: fused representation (DINOv2 local_box, PIE-trained baseline) ---')
    results['baseline_dinov2box_fused'] = run_probe(pie_repr_fused, jaad_repr_fused, label='baseline_dinov2box_fused')

    log.info('--- Probe: local_box branch hidden state (DINOv2) ---')
    results['baseline_dinov2box_branch'] = run_probe(
        pie_repr_box, jaad_repr_box, label='baseline_dinov2box_branch')

    out = os.path.join(RESULTS_DIR, 'dinov2_localbox_probe.pkl')
    with open(out, 'wb') as f:
        pickle.dump(results, f)
    log.info('Saved: %s', out)

    print('\n' + '=' * 90)
    print('DINOv2-backed local_box: separability probe (contrast case vs. local_context)')
    print('-' * 90)
    print(f'{"condition":<38}{"linear_acc":>12}{"+/-":>8}{"mlp_acc":>10}{"+/-":>8}')
    for k in ['baseline_dinov2box_fused', 'baseline_dinov2box_branch']:
        v = results[k]
        print(f'{k:<38}{v["acc_mean"]:>12.4f}{v["acc_std"]:>8.4f}{v["mlp_acc_mean"]:>10.4f}{v["mlp_acc_std"]:>8.4f}')
    print('-' * 90)
    print('VGG16 local_box baseline would need a dedicated probe run to compare directly')
    print('(this project\'s existing probes tap local_context and fused, not local_box, branch-wise).')
    print('=' * 90)


if __name__ == '__main__':
    main()
