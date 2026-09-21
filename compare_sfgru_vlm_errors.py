"""
compare_sfgru_vlm_errors.py
==============================
Checks whether SF-GRU's (trained PIE->JAAD, no-speed baseline) and the
VLM's (Qwen2-VL-2B zero-shot, see vlm_zeroshot_test.py) errors on the
SAME 60 JAAD test samples overlap. Two possible outcomes and what each
would mean:

  High overlap: both approaches fail on the same specific examples ->
    evidence those examples are intrinsically hard/ambiguous (occlusion,
    borderline intent, distant/small pedestrians) rather than either
    model having a specific, fixable weakness -- a different, more
    fundamental explanation than anything in the four architectural
    fixes (box exclusion, domain-adversarial, scale-norm,
    class-balance) tested this session.

  Low overlap: the two approaches fail on different examples -> their
    failure modes are genuinely distinct (e.g. SF-GRU overfits to PIE's
    box statistics; the VLM's weakness is unrelated to box at all,
    consistent with pose/box being SF-GRU-specific issues rather than
    intrinsic task difficulty).

Uses the exact same 60 JAAD test-set sample indices from
results/vlm_zeroshot_jaad.pkl, so the comparison is apples-to-apples
(same pedestrians, same ground truth), not a re-sample.

Usage
-----
  python compare_sfgru_vlm_errors.py

Output
------
  results/sfgru_vlm_error_overlap.pkl
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

SFGRU_DIR    = '/usr1/home/mehon/emma_pedestrian-intent-multimodal'
PIE_UTIL_DIR = '/usr1/home/mehon/PIE/utilities'
JAAD_UTIL_DIR = '/usr1/home/mehon/JAAD'
JAAD_DATA_DIR = '/usr1/home/mehon/JAAD'
RESULTS_DIR  = os.path.join(SFGRU_DIR, 'results')

sys.path.insert(0, SFGRU_DIR)
sys.path.append(PIE_UTIL_DIR)
sys.path.append(JAAD_UTIL_DIR)
os.chdir(SFGRU_DIR)

import train_full_pie_nospeed as _pie_mod
import train_full_jaad_nospeed as _jaad_mod

import sf_gru_torch as _sfgru_mod
from jaad_data import JAAD
from sf_gru_torch import SFGRUTorch


def main():
    vlm_data = pickle.load(open(os.path.join(RESULTS_DIR, 'vlm_zeroshot_jaad.pkl'), 'rb'))
    vlm_results = vlm_data['results']
    target_indices = [r['idx'] for r in vlm_results]
    log.info('Comparing on the exact %d JAAD test indices the VLM was evaluated on',
             len(target_indices))

    # Load SF-GRU's PIE->JAAD cross-dataset audit to get a trained checkpoint + threshold.
    cross_audit = pickle.load(open(os.path.join(RESULTS_DIR, 'cross_dataset_audit.pkl'), 'rb'))
    run = cross_audit['pie_to_jaad_runs'][0]  # seed 0
    model_path, threshold = run['model_path'], run['threshold']
    log.info('Using SF-GRU checkpoint: %s (threshold=%.2f)', model_path, threshold)

    # Generate JAAD test data the same way cross_dataset_eval.py does.
    _sfgru_mod.SFGRUTorch.get_pose = _jaad_mod._safe_get_pose_jaad
    _sfgru_mod.SFGRUTorch.load_images_crop_and_process = _jaad_mod._load_images_crop_and_process_jaad
    imdb = JAAD(data_path=JAAD_DATA_DIR)
    beh_test = imdb.generate_data_trajectory_sequence('test', **_jaad_mod.DATA_OPTS)

    method = SFGRUTorch()
    with open(os.path.join(model_path, 'model_opts.pkl'), 'rb') as fid:
        model_opts = pickle.load(fid)
    checkpoint = torch.load(os.path.join(model_path, 'model.pt'), map_location=method.device)
    model = method.build_model(checkpoint['data_types'], checkpoint['data_sizes'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    data, data_types, _ = method.get_data({'test': beh_test}, model_opts)
    raw_inputs, raw_labels = data['test']
    inputs = [torch.from_numpy(np.asarray(x)).float().to(method.device) for x in raw_inputs]
    labels = np.asarray(raw_labels).ravel()

    with torch.no_grad():
        sfgru_probs = model(inputs).squeeze(-1).cpu().numpy().ravel()
    sfgru_preds_all = (sfgru_probs >= threshold).astype(int)

    log.info('SF-GRU produced %d predictions total (full JAAD test set); '
             'matching down to the %d VLM-evaluated indices', len(sfgru_preds_all), len(target_indices))

    # Sanity check: label alignment. beh_test as generated here should be
    # the SAME order/length as what vlm_zeroshot_test.py indexed into,
    # since both call imdb.generate_data_trajectory_sequence('test', **DATA_OPTS)
    # with the identical DATA_OPTS -- verify labels match at each target index
    # before trusting the comparison.
    vlm_label_by_idx = {r['idx']: r['label'] for r in vlm_results}
    mismatches = 0
    for idx in target_indices:
        if idx >= len(labels):
            log.error('Index %d out of range for SF-GRU label array (len=%d) -- '
                     'data generation order/length mismatch, aborting', idx, len(labels))
            sys.exit(1)
        if int(labels[idx]) != vlm_label_by_idx[idx]:
            mismatches += 1
            log.warning('Label mismatch at idx=%d: SF-GRU says %d, VLM record says %d',
                       idx, int(labels[idx]), vlm_label_by_idx[idx])
    if mismatches:
        log.error('%d/%d label mismatches -- data ordering is NOT aligned between the two '
                 'runs; the comparison below is NOT trustworthy without resolving this.',
                 mismatches, len(target_indices))
    else:
        log.info('All %d labels match between SF-GRU data generation and the VLM run -- '
                'index alignment confirmed, comparison is valid.', len(target_indices))

    comparison = []
    for r in vlm_results:
        idx = r['idx']
        label = r['label']
        vlm_pred = r['pred']
        sfgru_pred = int(sfgru_preds_all[idx])
        vlm_correct = (vlm_pred == label) if vlm_pred != -1 else None
        sfgru_correct = (sfgru_pred == label)
        comparison.append({
            'idx': idx, 'label': label,
            'vlm_pred': vlm_pred, 'vlm_correct': vlm_correct,
            'sfgru_pred': sfgru_pred, 'sfgru_correct': sfgru_correct,
        })

    both_correct = sum(1 for c in comparison if c['vlm_correct'] and c['sfgru_correct'])
    both_wrong = sum(1 for c in comparison if c['vlm_correct'] is False and c['sfgru_correct'] is False)
    only_vlm_wrong = sum(1 for c in comparison if c['vlm_correct'] is False and c['sfgru_correct'] is True)
    only_sfgru_wrong = sum(1 for c in comparison if c['vlm_correct'] is True and c['sfgru_correct'] is False)
    vlm_unclear = sum(1 for c in comparison if c['vlm_correct'] is None)

    n_scored = len(comparison) - vlm_unclear
    sfgru_acc = sum(1 for c in comparison if c['sfgru_correct']) / len(comparison)
    vlm_acc = sum(1 for c in comparison if c['vlm_correct']) / n_scored if n_scored else float('nan')

    # Overlap statistic: of the samples EITHER approach gets wrong, what
    # fraction do BOTH get wrong? High -> shared difficulty; low -> distinct
    # failure modes.
    either_wrong = both_wrong + only_vlm_wrong + only_sfgru_wrong
    overlap_frac = both_wrong / either_wrong if either_wrong > 0 else float('nan')

    summary = {
        'n_samples': len(comparison), 'n_vlm_unclear': vlm_unclear,
        'sfgru_acc': sfgru_acc, 'vlm_acc': vlm_acc,
        'both_correct': both_correct, 'both_wrong': both_wrong,
        'only_vlm_wrong': only_vlm_wrong, 'only_sfgru_wrong': only_sfgru_wrong,
        'overlap_frac_of_either_wrong': overlap_frac,
        'label_alignment_mismatches': mismatches,
    }

    out = os.path.join(RESULTS_DIR, 'sfgru_vlm_error_overlap.pkl')
    with open(out, 'wb') as f:
        pickle.dump({'comparison': comparison, 'summary': summary}, f)
    log.info('Saved: %s', out)

    print('\n' + '=' * 70)
    print('SF-GRU (PIE->JAAD) vs. VLM (zero-shot) error overlap, same 60 JAAD samples')
    print('-' * 70)
    for k, v in summary.items():
        print(f'{k}: {v}')
    print('-' * 70)
    print(f'Of samples either approach got wrong, {overlap_frac:.1%} were wrong for BOTH')
    print('(chance-level overlap under independence would be roughly '
         f'(1-sfgru_acc)*(1-vlm_acc) / [(1-sfgru_acc)+(1-vlm_acc)-(1-sfgru_acc)*(1-vlm_acc)] '
         f'= {((1-sfgru_acc)*(1-vlm_acc)) / ((1-sfgru_acc)+(1-vlm_acc)-(1-sfgru_acc)*(1-vlm_acc)):.1%} '
         'for reference, treating errors as independent)')
    print('=' * 70)


if __name__ == '__main__':
    main()
