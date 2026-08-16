"""
train_set03_comparison.py
==========================
Controlled OpenPose-vs-RTMPose comparison, restricted to PIE set03 -- the
only set with real (non-synthetic) OpenPose keypoints available
(data/features/pie/poses/pose_set03.pkl, from the original SF-GRU authors).

This is the fair, apples-to-apples pose-estimator comparison: both train and
test data come from the same set03 videos (via a random pedestrian-level
split within set03), so neither pose backend has an unfair data-availability
advantage. Contrast with train_full_pie.py, which trains on the *standard*
PIE split (set01+02+04 train / set03 test) -- there, OpenPose data doesn't
exist for the training sets at all, so a full-split OpenPose run would train
on zeroed pose and isn't a meaningful comparison (see that script's --backend
none for the correct zeroed-pose ablation on the full split instead).

PyTorch reimplementation of the original train_rtmpose.py (which built on
the Keras/TF stack sf_gru.py, had hardcoded paths from the original dev
machine, no seed loop, and the same pose-lookup key bug fixed here and in
train_full_pie.py/sf_gru_torch.py).

Usage
-----
  python train_set03_comparison.py --seeds 3

Output
------
  results/set03_comparison_<backend>.pkl
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

SFGRU_DIR    = '/usr1/home/mehon/pedestrian-intent-multimodal'
PIE_UTIL_DIR = '/usr1/home/mehon/PIE/utilities'
PIE_DATA_DIR = '/usr1/home/mehon/data_root/pie'
RESULTS_DIR  = os.path.join(SFGRU_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

sys.path.insert(0, SFGRU_DIR)
sys.path.append(PIE_UTIL_DIR)
os.chdir(SFGRU_DIR)

import sf_gru_torch as _sfgru_mod
from pie_data import PIE
from sf_gru_torch import SFGRUTorch
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

# -- Restrict PIE to set03 only, for both train and test -----------------------
# generate_data_trajectory_sequence('train'/'val'/'test'/'all', ...) all
# resolve to set03; data_split_type='random' then does a genuine
# pedestrian-level random train/test split within set03.
_orig_get_image_set_ids = PIE._get_image_set_ids

def _set03_only(self, image_set):
    return ['set03']

PIE._get_image_set_ids = _set03_only

# -- get_pose with correct key construction for both OpenPose and RTMPose ------
# (same fix as train_full_pie.py / sf_gru_torch.py -- kept as an explicit local
# copy here since this script controls PIE_POSE_BACKEND per-call rather than
# once at process start.)
def _safe_get_pose(self, img_sequences, ped_ids, file_path, data_type='train'):
    import re
    preferred_backend = os.environ.get('PIE_POSE_BACKEND', 'openpose').strip().lower()
    set_poses_list = os.listdir(file_path)
    pose_files_by_set = {}
    for s in sorted(set_poses_list):
        match = re.match(r'^pose_(set\d+)(?:_(.+))?\.pkl$', s)
        if not match:
            continue
        set_id  = match.group(1)
        backend = (match.group(2) or 'openpose').lower()
        pose_files_by_set.setdefault(set_id, []).append((backend, s))

    set_poses = {}
    for set_id, candidates in pose_files_by_set.items():
        chosen = next(((b, f) for b, f in candidates if b == preferred_backend), None)
        if chosen is None:
            chosen = next(((b, f) for b, f in candidates if b == 'openpose'), None)
        if chosen is None:
            chosen = candidates[0]
        backend, file_name = chosen
        with open(os.path.join(file_path, file_name), 'rb') as fid:
            p = pickle.load(fid)
        set_poses[set_id] = p
    log.info('Pose sources: %s', {k: v[1] for k, v in
             {s: next(((b, f) for b, f in pose_files_by_set[s] if b == preferred_backend),
                      pose_files_by_set[s][0]) for s in pose_files_by_set}.items()})

    poses_all = []
    missing = 0
    total = 0
    for seq, pid in zip(img_sequences, ped_ids):
        pose = []
        for imp, p in zip(seq, pid):
            flip_image = False
            set_id   = imp.split('/')[-3]
            vid_id   = imp.split('/')[-2]
            img_name = imp.split('/')[-1].split('.')[0]
            if 'flip' in img_name:
                img_name = img_name.replace('_flip', '')
                flip_image = True
            total += 1
            vid_poses = set_poses.get(set_id, {}).get(vid_id, {})
            # OpenPose-style key: frame_<ped_id> (ped_id already includes
            # set/vid, e.g. '3_4_344' -> '01379_3_4_344').
            k = img_name + '_' + p[0]
            # RTMPose-style key (extract_rtmpose.py): extra set_num/vid_num
            # segment before ped_id, e.g. '01013_1_0001_1_1_1'.
            if k not in vid_poses:
                set_num = set_id.replace('set', '').lstrip('0') or '0'
                vid_num = vid_id.replace('video_', '')
                k_rtmpose = '%s_%s_%s_%s' % (img_name, set_num, vid_num, p[0])
                if k_rtmpose in vid_poses:
                    k = k_rtmpose
            if k in vid_poses:
                kp = vid_poses[k]
                pose.append(self.flip_pose(kp) if flip_image else kp)
            else:
                missing += 1
                pose.append([0] * 36)
        poses_all.append(pose)
    if total:
        log.info('Pose lookup: %d/%d frames missing (%.1f%%)', missing, total, 100.0 * missing / total)
    return np.array(poses_all)

_sfgru_mod.SFGRUTorch.get_pose = _safe_get_pose

# -- Patched get_path so features/models land under this repo's data/ tree -----
_FEAT_DIR = os.path.join(SFGRU_DIR, 'data', 'features')
_POSE_DIR = os.path.join(SFGRU_DIR, 'data', 'features', 'pie', 'poses')

def _patched_get_path(file_name='', save_folder='models', dataset='pie',
                      save_root_folder='data/'):
    if save_root_folder == 'data/features':
        save_path = (_POSE_DIR if save_folder == 'poses'
                     else os.path.join(_FEAT_DIR, dataset, save_folder))
    else:
        save_path = os.path.join(save_root_folder, dataset, save_folder)
    os.makedirs(save_path, exist_ok=True)
    return os.path.join(save_path, file_name), save_path

import utils as _u
_u.get_path = _patched_get_path
_sfgru_mod.get_path = _patched_get_path

DATA_OPTS = {
    'fstride': 1,
    'subset': 'default',
    'data_split_type': 'random',   # random pedestrian-level split within set03
    'seq_type': 'crossing',
    'min_track_size': 75,
}
MODEL_OPTS = {
    'obs_input_type': ['local_box', 'local_context', 'pose', 'box', 'speed'],
    'enlarge_ratio': 1.5,
    'pred_target_type': ['crossing'],
    'obs_length': 15,
    'time_to_event': 60,
    'dataset': 'pie',
    'normalize_boxes': True,
}


def summarise(runs):
    accs = np.array([r['acc'] for r in runs])
    aucs = np.array([r['auc'] for r in runs])
    f1s  = np.array([r['f1']  for r in runs])
    return {
        'acc_mean': accs.mean(), 'acc_std': accs.std(),
        'auc_mean': aucs.mean(), 'auc_std': aucs.std(),
        'f1_mean':  f1s.mean(),  'f1_std':  f1s.std(),
        'n': len(runs),
    }


def run_one_seed(seed, pose_backend):
    torch.manual_seed(seed)
    np.random.seed(seed)

    os.environ['PIE_POSE_BACKEND'] = pose_backend
    imdb = PIE(data_path=PIE_DATA_DIR)

    log.info('=== seed=%d  backend=%s ===', seed, pose_backend)

    # random split needs its own seed-consistent ratios; PIE caches the split
    # under data_cache using the ratios key, so re-running with the same
    # ratios (None -> default) but a different ratios-seed would normally
    # reuse the *same* cached split across seeds. To get a genuinely
    # different random train/test partition per seed (matching the intent
    # of "N independent trials"), pass regen_data=True with numpy's seed
    # already set above -- _get_random_pedestrian_ids reseeds via np.random.
    data_opts = dict(DATA_OPTS)
    data_opts['random_params'] = {'ratios': None, 'val_data': False, 'regen_data': True}

    beh_train = imdb.generate_data_trajectory_sequence('train', **data_opts)
    method = SFGRUTorch()

    saved_files_path = method.train(beh_train, model_opts=MODEL_OPTS)

    beh_test = imdb.generate_data_trajectory_sequence('test', **data_opts)
    acc, auc, f1, prec, rec = method.test(beh_test, saved_files_path)
    metrics = {'acc': acc, 'auc': auc, 'f1': f1, 'prec': prec, 'rec': rec,
               'seed': seed, 'backend': pose_backend}
    log.info('seed=%d  backend=%s  Acc=%.4f  AUC=%.4f  F1=%.4f', seed, pose_backend, acc, auc, f1)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--backend', choices=['openpose', 'rtmpose', 'both'], default='both',
                        help="Run just one backend (e.g. 'openpose' first to reproduce "
                             "the original paper's set03 numbers, then 'rtmpose' "
                             "separately to compare) or 'both' in one pass.")
    args = parser.parse_args()

    backends = ['openpose', 'rtmpose'] if args.backend == 'both' else [args.backend]

    # Load any existing comparison file so running backends separately
    # accumulates into the same results/set03_comparison.pkl instead of each
    # run clobbering the other's half.
    out = os.path.join(RESULTS_DIR, 'set03_comparison.pkl')
    all_results = {}
    if os.path.exists(out):
        with open(out, 'rb') as f:
            all_results = pickle.load(f)
        log.info('Loaded existing results for: %s', list(all_results.keys()))

    for backend in backends:
        runs = []
        for seed in range(args.seeds):
            m = run_one_seed(seed, backend)
            runs.append(m)
        summary = summarise(runs)
        all_results[backend] = {'runs': runs, 'summary': summary}
        log.info('\n%s (n=%d):  Acc %.4f+/-%.4f  AUC %.4f+/-%.4f  F1 %.4f+/-%.4f',
                 backend, summary['n'],
                 summary['acc_mean'], summary['acc_std'],
                 summary['auc_mean'], summary['auc_std'],
                 summary['f1_mean'],  summary['f1_std'])

        with open(out, 'wb') as f:
            pickle.dump(all_results, f)
        log.info('Saved: %s', out)

    print('\n' + '=' * 62)
    print(f'{"Backend":<14} {"Acc":>8} {"+/-":>5} {"AUC":>8} {"+/-":>5} {"F1":>8} {"+/-":>5}')
    print('-' * 62)
    for name, res in all_results.items():
        s = res['summary']
        print(f'{name:<14} {s["acc_mean"]:>8.4f} {s["acc_std"]:>5.4f} '
              f'{s["auc_mean"]:>8.4f} {s["auc_std"]:>5.4f} '
              f'{s["f1_mean"]:>8.4f} {s["f1_std"]:>5.4f}')
    print('=' * 62)


if __name__ == '__main__':
    main()
