"""
domain_leakage_probe_crossattn.py
=====================================
Second-architecture replication of domain_leakage_probe.py's central
finding (probe-measured domain separability INCREASES, not decreases,
for GRL/GRL+contrastive checkpoints relative to baseline, despite AUC
also increasing -- a dissociation between separability and task
performance) on CrossAttentionSFGRU's box-anchored cross-attention
fusion (sf_gru_torch_fusion.py / sf_gru_torch_domain_adversarial_contrastive_crossattn.py)
instead of the plain stacked-GRU fusion the original probe used.

Compares, for the fused representation (post-attention, post-final-GRU
hidden state -- the same tensor the crossing-intent head reads):
  - baseline (plain CrossAttentionSFGRU, no adaptation) -- reuses the
    checkpoint from full_pie_fusion_nospeed_cross_attn_rtmpose.pkl
    (already trained, see cross_dataset_fusion_audit_cross_attn_behonly.pkl)
  - GRL + contrastive (train_domain_adversarial_contrastive_crossattn.py,
    this project's cross-attention replication of the headline method)

Reuses domain_leakage_probe.py's run_probe() (linear + MLP probe, 5
repeats) and track-length trivial control unchanged -- only the
representation-extraction routine differs, since CrossAttentionStackedGRU's
forward pass (per-modality encoders -> cross-attention -> final GRU) is
structurally different from the plain stacked-GRU's sequential
concat-and-GRU forward pass domain_leakage_probe.py's extract_representation
assumes.

Usage
-----
  python domain_leakage_probe_crossattn.py

Output
------
  results/domain_leakage_probe_crossattn.pkl
"""

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

sys.path.insert(0, SFGRU_DIR)
sys.path.append(PIE_UTIL_DIR)
sys.path.append(JAAD_UTIL_DIR)
os.chdir(SFGRU_DIR)

from sf_gru_torch_fusion import CrossAttentionStackedGRU
from sf_gru_torch_domain_adversarial_contrastive_crossattn import (
    CrossAttentionDomainAdversarialContrastiveStackedGRU)

# Reuse the base probe script's data-loading, checkpoint-loading, and
# probe-fitting machinery unchanged -- only extract_representation needs
# a cross-attention-specific variant, defined below.
import domain_leakage_probe as _base


def extract_representation_crossattn(model, inputs, data_types, device='cuda'):
    """CrossAttention{,DomainAdversarialContrastive}StackedGRU's forward
    pass: per-modality GRU encoders (return_sequences=True) -> box-anchored
    cross-attention over the concatenated OTHER modalities' sequences ->
    one more GRU over the attended sequence -> final hidden state. This
    final hidden state (h.squeeze(0) after self.final_gru) is the fused
    representation the crossing-intent head reads -- the direct analogue
    of the plain stacked-GRU's 'fused' tap in domain_leakage_probe.py.
    Reimplements the forward pass up to that point rather than calling
    model.forward(), matching domain_leakage_probe.py's own convention of
    not relying on return_domain_logits=True plumbing for a probe that's
    deliberately independent of the model's own domain classifier."""
    model.eval()
    inputs_t = [torch.from_numpy(np.asarray(x)).float().to(device) for x in inputs]

    with torch.no_grad():
        sequences = []
        for gru, inp in zip(model.encoders, inputs_t):
            out, _ = gru(inp)
            sequences.append(out)

        anchor_seq = sequences[model.anchor_idx]
        others = [s for i, s in enumerate(sequences) if i != model.anchor_idx]
        context = torch.cat(others, dim=2)
        attended = model.cross_attn(anchor_seq, context)

        _, h = model.final_gru(attended)
        x = h.squeeze(0)

    return x.cpu().numpy()


def load_crossattn_checkpoint(model_path, model_class, hidden_units=256, anchor='box'):
    checkpoint = torch.load(os.path.join(model_path, 'model.pt'),
                            map_location='cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model_class(checkpoint['data_types'], checkpoint['data_sizes'],
                        hidden_units=hidden_units, anchor=anchor).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint['data_types']


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}

    log.info('=== Extracting PIE test features ===')
    pie_inputs, pie_types, pie_sizes, pie_track_len = _base.get_test_features('pie')
    log.info('=== Extracting JAAD test features ===')
    jaad_inputs, jaad_types, jaad_sizes, jaad_track_len = _base.get_test_features('jaad')
    assert pie_types == jaad_types, (pie_types, jaad_types)

    # --- Trivial control (identical to the base probe -- track length
    # alone carries no model-dependent information, so it does not need
    # to be re-derived per architecture; included here anyway so this
    # script's output stands alone as a complete comparison table). ---
    log.info('--- Trivial control: track length only ---')
    results['trivial_track_length'] = _base.run_probe(
        pie_track_len, jaad_track_len, label='trivial_track_length')

    # --- Baseline (no adaptation): CrossAttentionSFGRU fused representation ---
    log.info('--- Baseline (PIE-trained CrossAttentionSFGRU, no adaptation): fused representation ---')
    with open(os.path.join(RESULTS_DIR, 'full_pie_fusion_nospeed_cross_attn_rtmpose.pkl'), 'rb') as f:
        pie_baseline_paths = [r['model_path'] for r in pickle.load(f)['runs']]
    base_model, base_types = load_crossattn_checkpoint(
        pie_baseline_paths[0], CrossAttentionStackedGRU)
    pie_repr_base = extract_representation_crossattn(base_model, pie_inputs, pie_types, device=device)
    jaad_repr_base = extract_representation_crossattn(base_model, jaad_inputs, jaad_types, device=device)
    results['baseline_crossattn_fused'] = _base.run_probe(
        pie_repr_base, jaad_repr_base, label='baseline_crossattn_fused')

    # --- GRL + contrastive (this project's cross-attention replication
    # of the headline method), PIE->JAAD direction: fused representation ---
    log.info('--- GRL + contrastive, cross-attention, PIE->JAAD (dlw=1,gamma=10): fused representation ---')
    dac_path = ('data/models/domain_adversarial_contrastive_crossattn/'
                'domain_adversarial_contrastive_crossattn_pie_to_jaad-dlw1-gamma10-seed0')
    dac_model, dac_types = load_crossattn_checkpoint(
        dac_path, CrossAttentionDomainAdversarialContrastiveStackedGRU)
    pie_repr_dac = extract_representation_crossattn(dac_model, pie_inputs, pie_types, device=device)
    jaad_repr_dac = extract_representation_crossattn(dac_model, jaad_inputs, jaad_types, device=device)
    results['grl_contrastive_crossattn_fused'] = _base.run_probe(
        pie_repr_dac, jaad_repr_dac, label='grl_contrastive_crossattn_fused')

    # --- JAAD->PIE direction, added as a follow-up: Table~tab:crossattn
    # shows GRL + contrastive HURTS AUC in this direction (0.512->0.446),
    # the opposite sign from PIE->JAAD and from every stacked-GRU
    # direction. Only PIE->JAAD had been probed so far -- probing this
    # direction too tests whether the separability/performance
    # dissociation itself is direction-independent (probe accuracy still
    # rises even though AUC falls here) or whether the dissociation
    # pattern is itself direction-dependent (which would be a more
    # complex, and more interesting, finding than currently reported). ---
    log.info('--- Baseline (JAAD-trained CrossAttentionSFGRU, no adaptation): fused representation ---')
    with open(os.path.join(RESULTS_DIR, 'full_jaad_fusion_nospeed_behonly_cross_attn_rtmpose.pkl'), 'rb') as f:
        jaad_baseline_paths = [r['model_path'] for r in pickle.load(f)['runs']]
    base_model_jp, base_types_jp = load_crossattn_checkpoint(
        jaad_baseline_paths[0], CrossAttentionStackedGRU)
    pie_repr_base_jp = extract_representation_crossattn(base_model_jp, pie_inputs, pie_types, device=device)
    jaad_repr_base_jp = extract_representation_crossattn(base_model_jp, jaad_inputs, jaad_types, device=device)
    results['baseline_crossattn_fused_jaad_trained'] = _base.run_probe(
        pie_repr_base_jp, jaad_repr_base_jp, label='baseline_crossattn_fused_jaad_trained')

    log.info('--- GRL + contrastive, cross-attention, JAAD->PIE (dlw=1,gamma=10): fused representation ---')
    dac_path_jp = ('data/models/domain_adversarial_contrastive_crossattn/'
                   'domain_adversarial_contrastive_crossattn_jaad_to_pie-dlw1-gamma10-seed0')
    dac_model_jp, dac_types_jp = load_crossattn_checkpoint(
        dac_path_jp, CrossAttentionDomainAdversarialContrastiveStackedGRU)
    pie_repr_dac_jp = extract_representation_crossattn(dac_model_jp, pie_inputs, pie_types, device=device)
    jaad_repr_dac_jp = extract_representation_crossattn(dac_model_jp, jaad_inputs, jaad_types, device=device)
    results['grl_contrastive_crossattn_fused_jaad_to_pie'] = _base.run_probe(
        pie_repr_dac_jp, jaad_repr_dac_jp, label='grl_contrastive_crossattn_fused_jaad_to_pie')

    out = os.path.join(RESULTS_DIR, 'domain_leakage_probe_crossattn.pkl')
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
    print('Compare against domain_leakage_probe.py\'s stacked-GRU results: if')
    print('grl_contrastive_crossattn_fused is ALSO more separable than')
    print('baseline_crossattn_fused despite better cross-dataset AUC, the')
    print('separability/task-performance dissociation is not an SF-GRU-specific')
    print('artifact.')


if __name__ == '__main__':
    main()
