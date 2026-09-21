"""
train_domain_adversarial_contrastive_videogrouped.py
========================================================
Investigation #4 follow-up: the pedestrian-level random split
(train_domain_adversarial_contrastive_randomsplit.py) found a real AUC
drop vs. the default split (0.511+/-0.036 vs 0.618+/-0.019), but that
split has NO video-grouping constraint -- confirmed earlier that 98/188
JAAD test videos (52%) and 52/52 PIE test videos (100%) also contribute
pedestrians to their respective training sets. If video-level leakage
were INFLATING the default split's score, a leakage-FREE video-grouped
split should score similarly or lower, not higher, than the leaky
pedestrian-level one -- since removing a source of inflation should
make the number go DOWN, not up, if it was ever inflating anything.

This script uses PRE-GENERATED video-grouped split caches
(generate_video_grouped_split.py for JAAD, generate_video_grouped_split_pie.py
for PIE -- run those FIRST, this script does NOT regenerate or delete
them) -- data_split_type='random' with regen_data=False, so
_get_random_pedestrian_ids() reads the video-grouped cache file
verbatim instead of generating a fresh (non-grouped) split.

Usage
-----
  python train_domain_adversarial_contrastive_videogrouped.py --run_id 0
  (run generate_video_grouped_split.py and generate_video_grouped_split_pie.py
  FIRST, exactly once per desired split -- this script will use
  whatever video-grouped cache is currently on disk, unchanged.)

Output
------
  results/domain_adversarial_contrastive_videogrouped_pie_to_jaad_run<id>.pkl
"""

import argparse
import copy
import logging
import os
import sys

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

SFGRU_DIR = '/usr1/home/mehon/emma_pedestrian-intent-multimodal'
sys.path.insert(0, SFGRU_DIR)
os.chdir(SFGRU_DIR)

import train_domain_adversarial_contrastive as _base
import train_full_pie_nospeed as _pie_mod
import train_full_jaad_nospeed_behonly as _jaad_mod

PIE_RANDOM_CACHE = '/usr1/home/mehon/data_root/pie/data_cache/random_samples.pkl'
JAAD_RANDOM_CACHE = '/usr1/home/mehon/JAAD/data_cache/random_samples.pkl'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=int, required=True,
                        help='Arbitrary tag for this random-split run '
                             '(NOT a seed -- there is no seed control for '
                             'the split itself, only for model init/training, '
                             'this just labels the output file).')
    parser.add_argument('--source', default='pie', choices=['pie', 'jaad'])
    parser.add_argument('--target', default='jaad', choices=['pie', 'jaad'])
    args = parser.parse_args()

    # Do NOT delete or regenerate the random-split caches -- they must
    # already contain the VIDEO-GROUPED split produced by
    # generate_video_grouped_split.py / generate_video_grouped_split_pie.py.
    # regen_data=False here means _get_random_pedestrian_ids() reads
    # whatever is on disk verbatim; if either cache is missing, that
    # call will raise (rather than silently generating a fresh,
    # non-grouped split), which is the correct failure mode -- this
    # script has no business creating its own split.
    for cache_path in (PIE_RANDOM_CACHE, JAAD_RANDOM_CACHE):
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f'{cache_path} does not exist -- run generate_video_grouped_split.py '
                f'/ generate_video_grouped_split_pie.py first to create the '
                f'video-grouped split this script expects to find.')
        log.info('Using existing (expected video-grouped) cache: %s', cache_path)

    # Monkeypatch DATA_OPTS on both imported modules IN PROCESS for this
    # run only -- train_domain_adversarial_contrastive.py's get_raw_data()
    # reads _pie_mod.DATA_OPTS / _jaad_mod.DATA_OPTS by reference at call
    # time, so mutating them here before run_one_seed() is called is
    # sufficient; no need to touch the base training scripts, which stay
    # exactly as they are for the default-split figures already reported.
    for mod, val_data in [(_pie_mod, True), (_jaad_mod, True)]:
        mod.DATA_OPTS = copy.deepcopy(mod.DATA_OPTS)
        mod.DATA_OPTS['data_split_type'] = 'random'
        mod.DATA_OPTS['random_params'] = {
            'ratios': [0.5, 0.4, 0.1], 'val_data': val_data, 'regen_data': False,
        }
        log.info('%s DATA_OPTS patched: data_split_type=random, regen_data=False '
                 '(reads existing video-grouped cache)', mod.__name__)

    # Offset by 2000 so this script's checkpoint folder name
    # (domain_adversarial_contrastive_<source>_to_<target>-dlw1-gamma10-seed<N>)
    # can never collide with any real seed (0-5) used by the default-
    # split headline runs or the pedestrian-level randomsplit script
    # (which uses seed=run_id directly, i.e. 0-2 -- confirmed by
    # inspection that this script's own naming has no split-type tag,
    # only source/target/dlw/gamma/seed, so an unoffset seed here would
    # silently overwrite one of those checkpoints on disk).
    seed = 2000 + args.run_id
    run = _base.run_one_seed(seed, args.source, args.target, pose_backend='rtmpose')
    log.info('run_id=%d [%s->%s]: best_src_val_auc=%.4f', args.run_id, args.source, args.target,
             run['best_val_auc'])

    result = _base.evaluate_on_target_test(run, args.target, pose_backend='rtmpose')
    log.info('run_id=%d [%s->%s] RANDOM SPLIT: Acc=%.4f AUC=%.4f F1=%.4f',
             args.run_id, args.source, args.target, result['acc'], result['auc'], result['f1'])

    import pickle
    out = os.path.join(SFGRU_DIR, 'results',
                       f'domain_adversarial_contrastive_videogrouped_{args.source}_to_{args.target}_run{args.run_id}.pkl')
    with open(out, 'wb') as f:
        pickle.dump({'run_id': args.run_id, 'source': args.source, 'target': args.target,
                    'run': run, 'result': result}, f)
    log.info('Saved: %s', out)


if __name__ == '__main__':
    main()
