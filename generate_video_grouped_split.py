"""
generate_video_grouped_split.py
===================================
Investigation #4: is the random-split AUC drop (default-split GRL+contrastive
0.618+/-0.019 vs random-split 0.511+/-0.036, found earlier this session)
actually caused by video-level leakage, or by something else about the
random split's different/larger test-set composition? The random split
partitions individual PEDESTRIANS with no video-grouping constraint --
confirmed earlier that 98/188 JAAD test videos (52%) also contribute
pedestrians to the training set under that split. If video-level
leakage were inflating the DEFAULT split's score, a leakage-FREE random
split should score similarly to (or lower than) the leaky one, not
higher -- since removing leakage should make the task harder, not
easier, if leakage was ever inflating the number.

This script builds a JAAD-compatible random_samples.pkl-style split, but
GROUPS BY VIDEO first (splitting whole videos into train/val/test, then
pooling all of a video's qualifying pedestrians into whichever split its
video landed in) -- same ratios as jaad_data.py's default random split
([0.5, 0.4, 0.1]), same 'beh' sample_type filter, but with the video-
level leakage explicitly removed. This directly isolates whether video-
grouping (as opposed to the pedestrian-level pool size/composition) is
what's driving the observed AUC difference.

Does NOT modify jaad_data.py (a shared library file) -- instead
generates its own cache file in the same format
_get_random_pedestrian_ids() expects, so the existing training
infra (data_split_type='random', regen_data=False) picks it up
unmodified once this script's output is placed at JAAD's
data_cache/random_samples.pkl path.

Usage
-----
  python generate_video_grouped_split.py --run_id 0

Output
------
  Writes /usr1/home/mehon/JAAD/data_cache/random_samples.pkl directly
  (same path _get_random_pedestrian_ids() reads/writes) -- CALLER is
  responsible for deleting/backing up any existing file first, same
  as train_domain_adversarial_contrastive_randomsplit.py already does
  for the (pedestrian-level) random split.
"""

import argparse
import pickle
import sys

import numpy as np

sys.path.append('/usr1/home/mehon/JAAD')
from jaad_data import JAAD

JAAD_DATA_DIR = '/usr1/home/mehon/JAAD'
CACHE_PATH = '/usr1/home/mehon/JAAD/data_cache/random_samples.pkl'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=int, required=True,
                        help='Arbitrary tag for logging -- there is no seed control for '
                             'sklearn\'s train_test_split here either (matching the '
                             'pedestrian-level random split\'s own lack of seeding), so '
                             'this only labels which invocation produced the file, not a '
                             'reproducible split.')
    parser.add_argument('--sample_type', default='beh', choices=['beh', 'all'])
    parser.add_argument('--ratios', type=float, nargs=3, default=[0.5, 0.4, 0.1])
    args = parser.parse_args()

    imdb = JAAD(data_path=JAAD_DATA_DIR)
    annotations = imdb.generate_database()

    # Group qualifying pedestrian IDs by video, matching
    # _get_pedestrian_ids()'s own sample_type filter exactly.
    video_to_pids = {}
    for vid in sorted(annotations):
        if args.sample_type == 'beh':
            pids = [p for p in annotations[vid]['ped_annotations'].keys() if 'b' in p]
        else:
            pids = list(annotations[vid]['ped_annotations'].keys())
        if pids:
            video_to_pids[vid] = pids

    videos = sorted(video_to_pids.keys())
    print(f'Total qualifying videos: {len(videos)}, '
         f'total qualifying pedestrians: {sum(len(v) for v in video_to_pids.values())}')

    from sklearn.model_selection import train_test_split
    train_videos, testval_videos = train_test_split(videos, train_size=args.ratios[0])
    test_videos, val_videos = train_test_split(
        testval_videos, train_size=args.ratios[1] / sum(args.ratios[1:]))

    train_pids = [p for v in train_videos for p in video_to_pids[v]]
    val_pids = [p for v in val_videos for p in video_to_pids[v]]
    test_pids = [p for v in test_videos for p in video_to_pids[v]]

    print(f'run_id={args.run_id}: video-grouped split -- '
         f'train videos={len(train_videos)} ({len(train_pids)} peds), '
         f'val videos={len(val_videos)} ({len(val_pids)} peds), '
         f'test videos={len(test_videos)} ({len(test_pids)} peds)')

    # Sanity check: verify zero video overlap between splits (this is
    # the entire point of this script -- fail loudly if it's wrong).
    train_v, val_v, test_v = set(train_videos), set(val_videos), set(test_videos)
    assert not (train_v & test_v), 'BUG: train/test video overlap'
    assert not (train_v & val_v), 'BUG: train/val video overlap'
    assert not (val_v & test_v), 'BUG: val/test video overlap'
    print('Verified: zero video-level overlap between train/val/test.')

    sample_split = {
        'ratios': args.ratios,
        'train': train_pids,
        'val': val_pids,
        'test': test_pids,
    }
    with open(CACHE_PATH, 'wb') as f:
        pickle.dump(sample_split, f, pickle.HIGHEST_PROTOCOL)
    print(f'Saved video-grouped split to {CACHE_PATH}')


if __name__ == '__main__':
    main()
