"""
generate_video_grouped_split_pie.py
=======================================
PIE-side counterpart to generate_video_grouped_split.py. Same rationale
(video-level leakage under data_split_type='random' -- confirmed earlier
this session at 100% train/test video overlap for PIE's own random
split, 52/52 videos, even worse than JAAD's 52%). PIE's annotation
structure has an extra set_id level (annotations[sid][vid], vs JAAD's
flat annotations[vid]), so pedestrian grouping is keyed on (sid, vid)
pairs here, not vid alone.

Usage
-----
  python generate_video_grouped_split_pie.py --run_id 0

Output
------
  Writes /usr1/home/mehon/data_root/pie/data_cache/random_samples.pkl
  directly (same path _get_random_pedestrian_ids() reads/writes).
"""

import argparse
import pickle
import sys

sys.path.append('/usr1/home/mehon/PIE/utilities')
from pie_data import PIE

PIE_DATA_DIR = '/usr1/home/mehon/data_root/pie'
CACHE_PATH = '/usr1/home/mehon/data_root/pie/data_cache/random_samples.pkl'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=int, required=True)
    parser.add_argument('--ratios', type=float, nargs=3, default=[0.5, 0.4, 0.1])
    args = parser.parse_args()

    imdb = PIE(data_path=PIE_DATA_DIR)
    annotations = imdb.generate_database()

    video_to_pids = {}
    for sid in sorted(annotations):
        for vid in sorted(annotations[sid]):
            pids = list(annotations[sid][vid]['ped_annotations'].keys())
            if pids:
                video_to_pids[(sid, vid)] = pids

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
