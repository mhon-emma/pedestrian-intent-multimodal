"""
domain_leakage_probe.py
==========================
Investigation #2/#3/#5 (mechanistic follow-up to the domain-adversarial
result, train_domain_adversarial_behonly.py / train_domain_adversarial_context.py
/ train_domain_adversarial_contrastive.py): does the trained model's
REPRESENTATION actually become harder to tell PIE from JAAD, and is that
change localized or diffuse?

Method: for a given trained checkpoint, run PIE test + JAAD test inputs
through the model up to a chosen tap point (either the final fused
hidden state, matching what the crossing-intent head reads, or -- for
context-only models -- the local_context branch's own hidden state
specifically), collect those representations with a domain label
(PIE=0/JAAD=1), and fit a SEPARATE small probe -- BOTH a linear one
(logistic regression) and a non-linear one (2-layer MLP) -- to predict
domain from the representation. Probe test accuracy is the measurement:
50% = the representation carries no decodable dataset identity (genuine
invariance); 100% = fully separable (no invariance at all). The MLP
probe is investigation #1's direct follow-up: the original run (linear
probe only) found the full-GRL representation was MORE linearly
separable than baseline (0.962 vs 0.877), the opposite of the intended
effect. Before concluding domain-adversarial training achieves no
invariance at all, we need to rule out "a linear probe was just too
weak to find a real (non-linear) invariant structure" -- if the MLP
probe ALSO can't push accuracy down toward 0.5, that rules out the
weak-probe explanation and strengthens the "it's a generic regularizer,
not a representation fix" interpretation already reported.

This is deliberately a NEW, small classifier trained fresh on frozen
features -- not a re-report of the GRL's own in-training domain_acc,
which was trained adversarially and is not equivalent to "how separable
is the final representation," since that adversarial training was
jointly optimizing against changing features (a moving target). The
probe here trains on a FIXED, frozen final checkpoint's representation
-- the standard "linear probe" methodology for measuring what a
representation encodes.

Compares, for the SAME representation (final fused hidden state):
  - baseline (train_full_pie_nospeed.py / train_full_jaad_nospeed_behonly.py,
    no adaptation at all)
  - full-representation GRL (train_domain_adversarial_behonly.py, dlw=1/gamma=10)
  - context-only GRL (train_domain_adversarial_context.py) -- probed at
    BOTH the final fused state (to see if leakage persists downstream of
    the regularized branch) AND the local_context branch's own hidden
    state specifically (to directly check whether ITS output became
    invariant, isolating whether the narrow GRL achieved its immediate
    target even though the overall result underperformed the broad one)
  - GRL + contrastive (train_domain_adversarial_contrastive.py) -- the
    project's current best result (PIE->JAAD AUC 0.618+/-0.019 n=6) --
    added in this run to see whether ITS representation shows the same
    invariance paradox as the standalone GRL, or whether the contrastive
    term's extra regularization changes the picture
  - trivial control: track length only (a single scalar per sample) --
    establishes the floor for "how separable are PIE vs JAAD from a
    dataset-fingerprint statistic that carries zero pedestrian-behavior
    information," so probe accuracy on real features can be judged
    against a meaningful baseline rather than against bare chance.

Usage
-----
  python domain_leakage_probe.py

Output
------
  results/domain_leakage_probe.pkl
"""

import logging
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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
from sf_gru_torch import SFGRUTorch, StackedGRU
from sf_gru_torch_domain_adversarial import DomainAdversarialStackedGRU
from sf_gru_torch_domain_adversarial_context import ContextDomainAdversarialStackedGRU
from sf_gru_torch_domain_adversarial_contrastive import DomainAdversarialContrastiveStackedGRU
from sf_gru_torch_full_combo import FullComboStackedGRU

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


def get_test_features(dataset, pose_backend='rtmpose'):
    """Returns (inputs, data_types, data_sizes, track_lengths) for the
    dataset's TEST split, natural/unbalanced distribution -- same
    convention used everywhere else in this project for TEST splits."""
    apply_dataset_monkeypatches(dataset)
    if dataset == 'pie':
        imdb = PIE(data_path=PIE_DATA_DIR)
        os.environ['PIE_POSE_BACKEND'] = pose_backend
        beh_test = imdb.generate_data_trajectory_sequence('test', **_pie_mod.DATA_OPTS)
    else:
        imdb = JAAD(data_path=JAAD_DATA_DIR)
        os.environ['JAAD_POSE_BACKEND'] = pose_backend
        beh_test = imdb.generate_data_trajectory_sequence('test', **_jaad_mod.DATA_OPTS)

    method = SFGRUTorch(device='cuda' if torch.cuda.is_available() else 'cpu')
    model_opts = dict(MODEL_OPTS_BASE)
    # Reuse train_full_{pie,jaad}_nospeed[_behonly].py's own dataset cache
    # keys ('pie_nospeed' / 'jaad_nospeed_behonly') rather than inventing a
    # new one -- VGG local_context features are cached per (dataset key,
    # set_id, vid_id, frame), so a fresh key forces full VGG recomputation
    # for every frame from scratch (confirmed: ~10-20k cached files already
    # exist under the nospeed keys vs. an empty dir under a new key).
    model_opts['dataset'] = 'pie_nospeed' if dataset == 'pie' else 'jaad_nospeed_behonly'
    data, data_types, data_sizes = method.get_data({'test': beh_test}, model_opts)
    inputs, _labels = data['test']
    # Track length (number of non-padding frames) as the trivial-control
    # feature -- box's own raw sequence length dimension gives this
    # directly without touching pedestrian appearance/motion content.
    track_lengths = np.array([len(t) for t in beh_test['image']], dtype=np.float32).reshape(-1, 1)
    return inputs, data_types, data_sizes, track_lengths


@torch.no_grad()
def extract_representation(model, inputs, data_types, tap='fused', device='cuda'):
    """Runs `inputs` through `model`'s GRU stack up to the requested tap
    point. tap='fused': the final GRU's hidden state (same representation
    the crossing-intent head reads). tap='local_context': the
    local_context GRU's OWN hidden state (only valid for
    ContextDomainAdversarialStackedGRU-style models with that branch
    present) -- reimplements the forward pass rather than calling
    model.forward(return_domain_logits=True), since the DANN/base models
    don't expose a local_context-specific tap and we want one consistent
    extraction routine across all model types."""
    model.eval()
    inputs_t = [torch.from_numpy(np.asarray(x)).float().to(device) for x in inputs]

    x = None
    context_hidden = None
    context_idx = data_types.index('local_context') if 'local_context' in data_types else None
    for i, gru in enumerate(model.grus):
        is_last = (i == len(model.grus) - 1)
        if i == 0:
            seq_in = inputs_t[0]
        else:
            seq_in = torch.cat([x, inputs_t[i]], dim=2)
        out, h = gru(seq_in)
        if context_idx is not None and i == context_idx:
            context_hidden = h.squeeze(0)
        x = out if not is_last else h.squeeze(0)

    if tap == 'fused':
        return x.cpu().numpy()
    elif tap == 'local_context':
        if context_hidden is None:
            raise ValueError("data_types has no 'local_context' entry")
        return context_hidden.cpu().numpy()
    else:
        raise ValueError(tap)


def load_checkpoint_model(model_path, model_class, hidden_units=256):
    """StackedGRU's constructor takes a positional weight_decay arg (only
    used to build its optimizer, which we never touch here -- a probe-only
    load never calls .train()); DomainAdversarialStackedGRU/
    ContextDomainAdversarialStackedGRU's domain_hidden kwarg defaults to
    64 and doesn't need overriding, but they don't accept weight_decay at
    all. Build each with the args its own class actually declares rather
    than one shared call."""
    checkpoint = torch.load(os.path.join(model_path, 'model.pt'),
                            map_location='cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if model_class is StackedGRU:
        model = model_class(checkpoint['data_types'], checkpoint['data_sizes'],
                            hidden_units, 0.001).to(device)
    else:
        model = model_class(checkpoint['data_types'], checkpoint['data_sizes'],
                            hidden_units=hidden_units).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint['data_types']


def run_probe(pie_repr, jaad_repr, label='probe', n_repeats=5, seed_base=0):
    """Fits n_repeats independent train/test-split probes -- BOTH linear
    (logistic regression) and non-linear (small 2-layer MLP) -- on
    different random splits each time, returning mean+/-std test
    accuracy for each. A single split can be lucky/unlucky given the
    small sample sizes here (JAAD test ~117-276), so we don't trust one
    split. The MLP uses a small hidden layer (32 units) sized to the
    small sample count here (a bigger MLP would just memorize the
    train split) and early_stopping to avoid overfitting further."""
    X = np.concatenate([pie_repr, jaad_repr], axis=0)
    y = np.concatenate([np.zeros(len(pie_repr)), np.ones(len(jaad_repr))])

    linear_accs = []
    mlp_accs = []
    for r in range(n_repeats):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=seed_base + r, stratify=y)

        linear_clf = LogisticRegression(max_iter=2000, C=1.0)
        linear_clf.fit(X_train, y_train)
        linear_accs.append(accuracy_score(y_test, linear_clf.predict(X_test)))

        mlp_clf = MLPClassifier(hidden_layer_sizes=(32,), max_iter=2000,
                                early_stopping=True, random_state=seed_base + r)
        mlp_clf.fit(X_train, y_train)
        mlp_accs.append(accuracy_score(y_test, mlp_clf.predict(X_test)))

    linear_accs = np.array(linear_accs)
    mlp_accs = np.array(mlp_accs)
    log.info('%s: linear_probe_acc=%.4f +/- %.4f  mlp_probe_acc=%.4f +/- %.4f '
             '(n_repeats=%d, n_pie=%d, n_jaad=%d)',
             label, linear_accs.mean(), linear_accs.std(),
             mlp_accs.mean(), mlp_accs.std(), n_repeats, len(pie_repr), len(jaad_repr))
    return {'acc_mean': float(linear_accs.mean()), 'acc_std': float(linear_accs.std()),
            'mlp_acc_mean': float(mlp_accs.mean()), 'mlp_acc_std': float(mlp_accs.std()),
            'n_repeats': n_repeats, 'n_pie': len(pie_repr), 'n_jaad': len(jaad_repr)}


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}

    log.info('=== Extracting PIE test features ===')
    pie_inputs, pie_types, pie_sizes, pie_track_len = get_test_features('pie')
    log.info('=== Extracting JAAD test features ===')
    jaad_inputs, jaad_types, jaad_sizes, jaad_track_len = get_test_features('jaad')

    assert pie_types == jaad_types, (pie_types, jaad_types)

    # --- Trivial control: track length alone ---
    log.info('--- Trivial control: track length only ---')
    results['trivial_track_length'] = run_probe(pie_track_len, jaad_track_len,
                                                  label='trivial_track_length')

    # --- Baseline (no adaptation): fused representation ---
    # Use the base-class StackedGRU with a PIE-trained and a JAAD-trained
    # checkpoint SEPARATELY is not meaningful for a domain probe (a probe
    # needs ONE shared feature extractor run on both datasets) -- so for
    # the "no adaptation" condition we use the PIE-trained base checkpoint
    # as the shared feature extractor (arbitrary choice of which model;
    # what matters is whether ITS representation of JAAD inputs vs PIE
    # inputs is separable). This mirrors exactly what "evaluate cross-
    # dataset" already does -- same checkpoint, both datasets' inputs.
    log.info('--- Baseline (PIE-trained, no adaptation): fused representation ---')
    with open(os.path.join(RESULTS_DIR, 'full_pie_nospeed_results_rtmpose.pkl'), 'rb') as f:
        pie_baseline_paths = [r['model_path'] for r in pickle.load(f)['runs']]
    base_model, base_types = load_checkpoint_model(pie_baseline_paths[0], StackedGRU)
    pie_repr_base = extract_representation(base_model, pie_inputs, pie_types, tap='fused', device=device)
    jaad_repr_base = extract_representation(base_model, jaad_inputs, jaad_types, tap='fused', device=device)
    results['baseline_fused'] = run_probe(pie_repr_base, jaad_repr_base, label='baseline_fused')

    # --- Full-representation GRL: fused representation ---
    log.info('--- Full-representation GRL (dlw=1,gamma=10): fused representation ---')
    da_path = 'data/models/domain_adversarial_behonly/domain_adversarial_behonly_pie_to_jaad-dlw1-gamma10-seed0'
    da_model, da_types = load_checkpoint_model(da_path, DomainAdversarialStackedGRU)
    pie_repr_da = extract_representation(da_model, pie_inputs, pie_types, tap='fused', device=device)
    jaad_repr_da = extract_representation(da_model, jaad_inputs, jaad_types, tap='fused', device=device)
    results['full_grl_fused'] = run_probe(pie_repr_da, jaad_repr_da, label='full_grl_fused')

    # --- Context-only GRL: fused representation AND local_context branch ---
    log.info('--- Context-only GRL: fused representation ---')
    ctx_path = 'data/models/domain_adversarial_context/domain_adversarial_context_pie_to_jaad-dlw1-gamma10-seed0'
    ctx_model, ctx_types = load_checkpoint_model(ctx_path, ContextDomainAdversarialStackedGRU)
    pie_repr_ctx_fused = extract_representation(ctx_model, pie_inputs, pie_types, tap='fused', device=device)
    jaad_repr_ctx_fused = extract_representation(ctx_model, jaad_inputs, jaad_types, tap='fused', device=device)
    results['context_grl_fused'] = run_probe(pie_repr_ctx_fused, jaad_repr_ctx_fused, label='context_grl_fused')

    log.info('--- Context-only GRL: local_context branch hidden state ---')
    pie_repr_ctx_branch = extract_representation(ctx_model, pie_inputs, pie_types, tap='local_context', device=device)
    jaad_repr_ctx_branch = extract_representation(ctx_model, jaad_inputs, jaad_types, tap='local_context', device=device)
    results['context_grl_local_context_branch'] = run_probe(
        pie_repr_ctx_branch, jaad_repr_ctx_branch, label='context_grl_local_context_branch')

    # --- For comparison: baseline model's local_context branch (was it
    # already leaking pre-intervention?) ---
    log.info('--- Baseline (no adaptation): local_context branch hidden state ---')
    pie_repr_base_branch = extract_representation(base_model, pie_inputs, pie_types, tap='local_context', device=device)
    jaad_repr_base_branch = extract_representation(base_model, jaad_inputs, jaad_types, tap='local_context', device=device)
    results['baseline_local_context_branch'] = run_probe(
        pie_repr_base_branch, jaad_repr_base_branch, label='baseline_local_context_branch')

    # --- GRL + contrastive (current project-best method): fused
    # representation -- does the added contrastive regularization change
    # the invariance-paradox picture, or does it show the same pattern
    # (more separable, not less, despite the AUC improvement)? ---
    log.info('--- GRL + contrastive (dlw=1,gamma=10): fused representation ---')
    dac_path = 'data/models/domain_adversarial_contrastive/domain_adversarial_contrastive_pie_to_jaad-dlw1-gamma10-seed0'
    dac_model, dac_types = load_checkpoint_model(dac_path, DomainAdversarialContrastiveStackedGRU)
    pie_repr_dac = extract_representation(dac_model, pie_inputs, pie_types, tap='fused', device=device)
    jaad_repr_dac = extract_representation(dac_model, jaad_inputs, jaad_types, tap='fused', device=device)
    results['grl_contrastive_fused'] = run_probe(pie_repr_dac, jaad_repr_dac, label='grl_contrastive_fused')

    # --- GRL + contrastive + mixup (same architecture as grl_contrastive,
    # only the SOURCE training data differs -- same-class mixup added
    # synthetic examples). Confirmed a net-negative AUC result (0.580 vs
    # 0.618 PIE->JAAD, with variance blowing up 4x); this checks whether
    # that mean/variance regression shows up as a corresponding change in
    # representation separability, or whether the representation looks
    # similar to grl_contrastive_fused despite the worse task performance
    # (which would mean mixup's damage is in the loss landscape /
    # optimization, not in what the representation ends up encoding). ---
    log.info('--- GRL + contrastive + mixup (dlw=1,gamma=10): fused representation ---')
    mixup_path = 'data/models/domain_adversarial_contrastive_mixup/domain_adversarial_contrastive_mixup_pie_to_jaad-dlw1-gamma10-seed0'
    mixup_model, mixup_types = load_checkpoint_model(mixup_path, DomainAdversarialContrastiveStackedGRU)
    pie_repr_mixup = extract_representation(mixup_model, pie_inputs, pie_types, tap='fused', device=device)
    jaad_repr_mixup = extract_representation(mixup_model, jaad_inputs, jaad_types, tap='fused', device=device)
    results['grl_contrastive_mixup_fused'] = run_probe(pie_repr_mixup, jaad_repr_mixup, label='grl_contrastive_mixup_fused')

    out = os.path.join(RESULTS_DIR, 'domain_leakage_probe.pkl')
    with open(out, 'wb') as f:
        pickle.dump(results, f)
    log.info('Saved: %s', out)

    print('\n' + '=' * 96)
    print(f'{"condition":<38}{"linear_acc":>12}{"+/-":>8}{"mlp_acc":>10}{"+/-":>8}{"n_pie":>8}{"n_jaad":>8}')
    print('-' * 96)
    for k, v in results.items():
        print(f'{k:<38}{v["acc_mean"]:>12.4f}{v["acc_std"]:>8.4f}'
              f'{v["mlp_acc_mean"]:>10.4f}{v["mlp_acc_std"]:>8.4f}{v["n_pie"]:>8}{v["n_jaad"]:>8}')
    print('=' * 96)
    print('50% = no decodable dataset identity (genuine invariance)')
    print('100% = fully separable (no invariance at all)')
    print('If mlp_acc << linear_acc for a GRL condition, the representation has real')
    print('invariant structure a linear probe missed. If mlp_acc stays high too, the')
    print('"no genuine invariance" conclusion holds under a stronger probe.')


if __name__ == '__main__':
    main()
