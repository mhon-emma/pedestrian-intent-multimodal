"""
train_domain_adversarial_contrastive_crossattn_videogrouped.py
====================================================================
Extends the video-grouped-split leakage check
(train_domain_adversarial_contrastive_videogrouped.py, which found
video-level leakage does NOT explain the default-split-vs-random-split
AUC gap on stacked-GRU -- the leakage-free video-grouped split scored
LOWER, not higher, than the leaky pedestrian-level random split) to
the cross-attention architecture, closing a combinatorial gap between
two existing findings that haven't been tested together: "the fix is
architecture-dependent" (Table tab:crossattn) and "video-level leakage
doesn't explain split-sensitivity" (stacked-GRU only, Section
results-leakage).

Requires PRE-GENERATED video-grouped split caches (same ones
train_domain_adversarial_contrastive_videogrouped.py uses --
generate_video_grouped_split.py / generate_video_grouped_split_pie.py;
this script does NOT regenerate them, same convention).

This is a thin wrapper around train_domain_adversarial_contrastive_crossattn.py's
run_one_seed/evaluate_on_target_test, with DATA_OPTS monkeypatched to
read the video-grouped split cache instead of the default split --
exactly mirroring the stacked-GRU video-grouped script's own pattern
(copy_pie_mod/_jaad_mod.DATA_OPTS, data_split_type='random',
regen_data=False).

Usage
-----
  python train_domain_adversarial_contrastive_crossattn_videogrouped.py --run_id 0 --source pie --target jaad
  (run generate_video_grouped_split.py and generate_video_grouped_split_pie.py
  FIRST, exactly once -- this script uses whatever video-grouped cache
  is currently on disk, unchanged.)

Output
------
  results/domain_adversarial_contrastive_crossattn_videogrouped_<source>_to_<target>_run<id>.pkl
"""

import argparse
import copy
import logging
import os
import pickle
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

SFGRU_DIR = '/usr1/home/mehon/emma_pedestrian-intent-multimodal'
sys.path.insert(0, SFGRU_DIR)
os.chdir(SFGRU_DIR)

import train_domain_adversarial_contrastive_crossattn as _base
import train_full_pie_nospeed as _pie_mod
import train_full_jaad_nospeed_behonly as _jaad_mod

PIE_RANDOM_CACHE = '/usr1/home/mehon/data_root/pie/data_cache/random_samples.pkl'
JAAD_RANDOM_CACHE = '/usr1/home/mehon/JAAD/data_cache/random_samples.pkl'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=int, required=True,
                        help='Arbitrary tag for this video-grouped-split run '
                             '(NOT a seed -- the split itself has no seed control, '
                             'only model init/training does; this just labels the '
                             'output file).')
    parser.add_argument('--source', default='pie', choices=['pie', 'jaad'])
    parser.add_argument('--target', default='jaad', choices=['pie', 'jaad'])
    args = parser.parse_args()

    for cache_path in (PIE_RANDOM_CACHE, JAAD_RANDOM_CACHE):
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f'{cache_path} does not exist -- run generate_video_grouped_split.py '
                f'/ generate_video_grouped_split_pie.py first to create the '
                f'video-grouped split this script expects to find.')
        log.info('Using existing (expected video-grouped) cache: %s', cache_path)

    for mod, val_data in [(_pie_mod, True), (_jaad_mod, True)]:
        mod.DATA_OPTS = copy.deepcopy(mod.DATA_OPTS)
        mod.DATA_OPTS['data_split_type'] = 'random'
        mod.DATA_OPTS['random_params'] = {
            'ratios': [0.5, 0.4, 0.1], 'val_data': val_data, 'regen_data': False,
        }
        log.info('%s DATA_OPTS patched: data_split_type=random, regen_data=False '
                 '(reads existing video-grouped cache)', mod.__name__)

    # Offset by 3000 -- distinct from the stacked-GRU video-grouped
    # script's 2000 offset and from the plain crossattn headline runs'
    # seeds 0-5, so this script's checkpoint folder name
    # (domain_adversarial_contrastive_crossattn_<source>_to_<target>-...-seed<N>)
    # can never collide with either.
    seed = 3000 + args.run_id
    run = _base.run_one_seed(seed, args.source, args.target, pose_backend='rtmpose')
    log.info('run_id=%d [%s->%s crossattn]: best_src_val_auc=%.4f', args.run_id, args.source, args.target,
             run['best_val_auc'])

    result = _base.evaluate_on_target_test(run, args.target, pose_backend='rtmpose')
    log.info('run_id=%d [%s->%s crossattn] VIDEO-GROUPED SPLIT: Acc=%.4f AUC=%.4f F1=%.4f',
             args.run_id, args.source, args.target, result['acc'], result['auc'], result['f1'])

    out = os.path.join(SFGRU_DIR, 'results',
                       f'domain_adversarial_contrastive_crossattn_videogrouped_{args.source}_to_{args.target}_run{args.run_id}.pkl')
    with open(out, 'wb') as f:
        pickle.dump({'run_id': args.run_id, 'source': args.source, 'target': args.target,
                    'run': run, 'result': result}, f)
    log.info('Saved: %s', out)


if __name__ == '__main__':
    main()
