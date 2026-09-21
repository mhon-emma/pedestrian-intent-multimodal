"""
train_domain_adversarial_contrastive_randomsplit.py
======================================================
Investigation #2 (split-robustness check): does the project's headline
result (GRL + contrastive, train_domain_adversarial_contrastive.py,
PIE->JAAD AUC 0.618+/-0.019 n=6) hold under a DIFFERENT train/val/test
partition, or is it specific to the literature-standard fixed split
(data_split_type='default': PIE train=set01+02+04/val=set05+06/test=set03,
JAAD's default/{train,val,test}.txt video-ID lists)?

Originally scoped as a k-fold check, but pie_data.py/jaad_data.py's
kfold_params is DEAD CONFIGURATION here -- every script in this project
uses data_split_type='default', so kfold_params (visible in every raw-data
generation log) was never actually consulted. There is no "fold 2, fold 3"
to test under the existing setup. data_split_type='random' is the actual
alternative both datasets support: a random per-pedestrian train/val/test
partition (ratios [0.5, 0.4, 0.1] by default, no val split disabled),
cached to disk (pie: data_root/pie/data_cache/random_samples.pkl; jaad:
similar) and NOT reproducible via a seed argument -- _get_random_pedestrian_ids
has no seed parameter at all (uses sklearn's train_test_split with no
random_state, i.e. numpy's global RNG state). regen_data=True forces a
FRESH random split each call; without it, the cached split from any
earlier call is silently reused. This script always passes
regen_data=True and deletes the datasets' own kfold/random split caches
before each run, so each invocation gets an independently fresh random
partition (not a controlled, reproducible one -- there is no seed to
control here, which is a real limitation of this check, not an oversight).

sample_type='beh' (the labeling-bug fix) is preserved: jaad_data.py
threads params['sample_type'] into random_params automatically
(_get_data via params['random_params']['sample_type'] = params['sample_type']
before calling _get_random_pedestrian_ids), so the random split still
correctly restricts to behavior-annotated JAAD pedestrians.

This trains ONE run of GRL+contrastive under a random split and
evaluates cross-dataset -- run this script multiple times (each one
regenerates a fresh random split) to build up a small distribution of
AUCs under different partitions, compared against the 0.618+/-0.019
default-split result.

Usage
-----
  python train_domain_adversarial_contrastive_randomsplit.py --run_id 0

Output
------
  results/domain_adversarial_contrastive_randomsplit_pie_to_jaad_run<id>.pkl
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

    # Force a fresh random split for both datasets -- delete any stale
    # cache first (belt-and-suspenders alongside regen_data=True below;
    # the cache file's own ratio-mismatch assertion would otherwise
    # raise if a previous run used different ratios, so starting clean
    # avoids that entirely).
    for cache_path in (PIE_RANDOM_CACHE, JAAD_RANDOM_CACHE):
        if os.path.exists(cache_path):
            os.remove(cache_path)
            log.info('Removed stale random-split cache: %s', cache_path)

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
            'ratios': [0.5, 0.4, 0.1], 'val_data': val_data, 'regen_data': True,
        }
        log.info('%s DATA_OPTS patched: data_split_type=random, ratios=[0.5,0.4,0.1]',
                 mod.__name__)

    # Offset by 1000 so this script's checkpoint folder name
    # (domain_adversarial_contrastive_<source>_to_<target>-dlw1-gamma10-seed<N>,
    # which carries no split-type tag) can never collide with the
    # default-split headline runs' checkpoints (seeds 0-5) -- a latent
    # bug found and fixed while building the video-grouped-split sibling
    # script; confirmed by checkpoint-file timestamps that this script's
    # prior runs never actually hit the collision, but it was one seed
    # choice away from silently overwriting seed0-2's checkpoints.
    seed = 1000 + args.run_id
    run = _base.run_one_seed(seed, args.source, args.target, pose_backend='rtmpose')
    log.info('run_id=%d [%s->%s]: best_src_val_auc=%.4f', args.run_id, args.source, args.target,
             run['best_val_auc'])

    result = _base.evaluate_on_target_test(run, args.target, pose_backend='rtmpose')
    log.info('run_id=%d [%s->%s] RANDOM SPLIT: Acc=%.4f AUC=%.4f F1=%.4f',
             args.run_id, args.source, args.target, result['acc'], result['auc'], result['f1'])

    import pickle
    out = os.path.join(SFGRU_DIR, 'results',
                       f'domain_adversarial_contrastive_randomsplit_{args.source}_to_{args.target}_run{args.run_id}.pkl')
    with open(out, 'wb') as f:
        pickle.dump({'run_id': args.run_id, 'source': args.source, 'target': args.target,
                    'run': run, 'result': result}, f)
    log.info('Saved: %s', out)


if __name__ == '__main__':
    main()
