"""
train_full_pie_heightnorm.py
==============================
Same as train_full_pie_nospeed.py, but uses HeightNormSFGRUTorch
(sf_gru_torch_heightnorm.py) instead of SFGRUTorch: each FRAME's box
position is rescaled by a depth-proportional factor implied by that
frame's own box height (assuming a standard ~1.7m adult), before taking
the frame-to-frame delta -- a per-frame perspective correction, unlike
sf_gru_torch_scalenorm.py's fixed single first-frame divisor. Sixth fix
attempt for the cross-dataset transfer failure diagnosed in
cross_dataset_modality_ablation.py; motivated by camera-calibration
research finding PIE has official calibration but JAAD does not and
can't have one uniform calibration (2 vehicles, 3 camera models, 5
locations) -- this height-based proxy needs no per-dataset calibration
at all. See sf_gru_torch_heightnorm.py's docstring for the full
motivation and derivation.

Standard PIE split
------------------
  train : set01, set02, set04
  val   : set05, set06
  test  : set03

Usage
-----
  python train_full_pie_heightnorm.py --seeds 3 --backend rtmpose

Output
------
  results/full_pie_heightnorm_results_<backend>.pkl
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

SFGRU_DIR    = '/usr1/home/mehon/emma_pedestrian-intent-multimodal'
PIE_UTIL_DIR = '/usr1/home/mehon/PIE/utilities'
PIE_DATA_DIR = '/usr1/home/mehon/data_root/pie'
RESULTS_DIR  = os.path.join(SFGRU_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

sys.path.insert(0, SFGRU_DIR)
sys.path.append(PIE_UTIL_DIR)
os.chdir(SFGRU_DIR)

import sf_gru_torch as _sfgru_mod
from pie_data import PIE
from sf_gru_torch_heightnorm import HeightNormSFGRUTorch
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

# -- get_pose with graceful zero-fallback on missing sets/frames ---------------
# (identical to train_full_pie_nospeed.py's version; patched onto the
# shared SFGRUTorch base class, so HeightNormSFGRUTorch inherits it too.)
def _safe_get_pose(self, img_sequences, ped_ids, file_path, data_type='train'):
    import re
    preferred_backend = os.environ.get('PIE_POSE_BACKEND', 'rtmpose').strip().lower()
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
            k = img_name + '_' + p[0]
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
    'data_split_type': 'default',   # train=set01+02+04, val=set05+06, test=set03
    'seq_type': 'crossing',
    'min_track_size': 75,
}
MODEL_OPTS = {
    'obs_input_type': ['local_box', 'local_context', 'pose', 'box'],  # no speed
    'enlarge_ratio': 1.5,
    'pred_target_type': ['crossing'],
    'obs_length': 15,
    'time_to_event': 60,
    'dataset': 'pie_heightnorm',  # distinct cache from pie_nospeed -- box values differ
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


def run_one_seed(seed, pose_backend, model_save_name, experiment_label=None, lr=3e-5):
    if experiment_label is None:
        experiment_label = pose_backend
    torch.manual_seed(seed)
    np.random.seed(seed)

    os.environ['PIE_POSE_BACKEND'] = pose_backend
    imdb = PIE(data_path=PIE_DATA_DIR)

    log.info('=== seed=%d  experiment=%s (pose_cache=%s) ===', seed, experiment_label, pose_backend)

    beh_train = imdb.generate_data_trajectory_sequence('train', **DATA_OPTS)
    beh_val   = imdb.generate_data_trajectory_sequence('val', **DATA_OPTS)
    method = HeightNormSFGRUTorch()

    saved_files_path = method.train(
        beh_train,
        data_val=beh_val,
        batch_size=32,
        epochs=100,
        lr=lr,
        model_opts=MODEL_OPTS
    )

    best_threshold = method.find_best_threshold(
        beh_val,
        saved_files_path,
        model_opts=MODEL_OPTS
    )

    beh_test = imdb.generate_data_trajectory_sequence('test', **DATA_OPTS)
    acc, auc, f1, prec, rec = method.test(beh_test, saved_files_path, threshold=best_threshold)
    metrics = {'acc': acc, 'auc': auc, 'f1': f1, 'prec': prec, 'rec': rec, 'best_threshold': best_threshold,
               'seed': seed, 'backend': experiment_label, 'model_path': saved_files_path}
    log.info('seed=%d  experiment=%s  Acc=%.4f  AUC=%.4f  F1=%.4f', seed, experiment_label, acc, auc, f1)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=3,
                        help='Number of independent training runs (default 3)')
    parser.add_argument('--backend', default='rtmpose',
                        choices=['rtmpose', 'openpose', 'none'],
                        help='Pose backend (default: rtmpose). "none" zeroes '
                             'the pose input for the ablation baseline.')
    parser.add_argument('--lr', type=float, default=3e-5,
                        help='Learning rate')
    args = parser.parse_args()

    if args.backend == 'none':
        orig_get_pose = _sfgru_mod.SFGRUTorch.get_pose
        def _zero_pose(self, img_sequences, ped_ids, file_path, data_type='train'):
            poses = orig_get_pose(self, img_sequences, ped_ids, file_path, data_type)
            return np.zeros_like(poses)
        _sfgru_mod.SFGRUTorch.get_pose = _zero_pose
        pose_env_backend = 'rtmpose'
    else:
        pose_env_backend = args.backend

    runs = []
    for seed in range(args.seeds):
        m = run_one_seed(seed, pose_env_backend, f'sf-rnn-{args.backend}-heightnorm',
                         experiment_label=args.backend, lr=args.lr)
        runs.append(m)

    summary = summarise(runs)
    log.info('\n%s (n=%d):  Acc %.4f+/-%.4f  AUC %.4f+/-%.4f  F1 %.4f+/-%.4f',
             args.backend, summary['n'],
             summary['acc_mean'], summary['acc_std'],
             summary['auc_mean'], summary['auc_std'],
             summary['f1_mean'],  summary['f1_std'])

    out = os.path.join(RESULTS_DIR, f'full_pie_heightnorm_results_{args.backend}.pkl')
    with open(out, 'wb') as f:
        pickle.dump({'runs': runs, 'summary': summary}, f)
    log.info('Saved: %s', out)

    print('\n' + '=' * 62)
    print(f'{"Backend":<14} {"Acc":>8} {"+/-":>5} {"AUC":>8} {"+/-":>5} {"F1":>8} {"+/-":>5}')
    print('-' * 62)
    print(f'{args.backend:<14} {summary["acc_mean"]:>8.4f} {summary["acc_std"]:>5.4f} '
          f'{summary["auc_mean"]:>8.4f} {summary["auc_std"]:>5.4f} '
          f'{summary["f1_mean"]:>8.4f} {summary["f1_std"]:>5.4f}')
    print('=' * 62)


if __name__ == '__main__':
    main()
