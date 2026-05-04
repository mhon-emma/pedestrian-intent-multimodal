"""
train_full_pie.py
=================
Trains SF-GRU on the standard PIE split using RTMPose poses and evaluates
on the held-out test set.  Runs N_SEEDS independent trials to report
mean ± std, matching the evaluation protocol of the original SF-GRU paper.

Standard PIE split
------------------
  train : set01, set02, set04   (pose available: RTMPose ✓  OpenPose ✗)
  val   : set05, set06          (pose available: RTMPose ✓  OpenPose ✗)
  test  : set03                 (pose available: RTMPose ✓  OpenPose ✓)

Note: OpenPose poses only exist for set03, so an OpenPose baseline on the
full split would train with zero poses — that is not a fair comparison.
See train_rtmpose.py for the within-set03 RTMPose vs OpenPose comparison.

Usage
-----
  /home/teamj/miniconda3/envs/sfgru_eval/bin/python -u train_full_pie.py \
      [--seeds 5] [--backend rtmpose] [--no-pose-baseline]

Output
------
  results/full_pie_results.pkl   — dict with per-seed metrics + summary
"""

import argparse
import os
import sys
import pickle
import types
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
SFGRU_DIR    = '/home/teamj/Documents/MMML/pedestrian-intent-multimodal'
PIE_UTIL_DIR = '/home/teamj/Documents/MMML/PIE/utilities'
PIE_DATA_DIR = '/home/teamj/Documents/MMML/PIE_data'
RESULTS_DIR  = os.path.join(SFGRU_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

sys.path.insert(0, SFGRU_DIR)
sys.path.append(PIE_UTIL_DIR)
os.chdir(SFGRU_DIR)

# ── Keras compatibility patches ────────────────────────────────────────────────
import keras
import keras.layers as kl
import keras.regularizers as kr

_rec = types.ModuleType('keras.layers.recurrent')
_rec.GRU = kl.GRU
sys.modules['keras.layers.recurrent'] = _rec
_core = types.ModuleType('keras.layers.core')
_core.regularizers = kr
sys.modules['keras.layers.core'] = _core

import utils as _utils_mod

def _jitter_bbox_no_imread(img_path, bbox, mode, ratio):
    """Drop-in replacement — avoids load_img; PIE frames are always 1920×1080."""
    from utils import bbox_sanity_check
    if mode == 'same':
        return bbox
    jitter_ratio = abs(ratio) if mode in ['random_enlarge', 'enlarge'] else ratio
    if mode == 'random_enlarge':
        jitter_ratio = np.random.random_sample() * jitter_ratio
    elif mode == 'random_move':
        jitter_ratio = np.random.random_sample() * jitter_ratio * 2 - jitter_ratio
    img_size = (1920, 1080)
    jit_boxes = []
    for b in bbox:
        bw = b[2] - b[0]; bh = b[3] - b[1]
        wc = bw * jitter_ratio; hc = bh * jitter_ratio
        if wc < hc: hc = wc
        else:       wc = hc
        if mode in ['enlarge', 'random_enlarge']:
            b[0] -= wc // 2; b[1] -= hc // 2
        else:
            b[0] += wc // 2; b[1] += hc // 2
        b[2] += wc // 2; b[3] += hc // 2
        b = bbox_sanity_check(img_size, b)
        jit_boxes.append(b)
    return jit_boxes

_utils_mod.jitter_bbox = _jitter_bbox_no_imread
import sf_gru as _sfgru_mod
_sfgru_mod.jitter_bbox = _jitter_bbox_no_imread

# ── Video-backed image loader (no PNG extraction needed) ───────────────────────
import cv2 as _cv2
from PIL import Image as _PIL_Image
from keras.preprocessing.image import img_to_array as _img_to_array
from keras.applications import vgg16 as _vgg16_mod

_convnet = None

def _get_convnet():
    global _convnet
    if _convnet is None:
        _convnet = _vgg16_mod.VGG16(input_shape=(224, 224, 3),
                                    include_top=False, weights='imagenet')
    return _convnet

def _frame_from_video(set_id, vid_id, frame_num):
    video_path = os.path.join('/home/teamj/Documents/MMML/Data', set_id, vid_id + '.mp4')
    cap = _cv2.VideoCapture(video_path)
    cap.set(_cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, bgr = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f'Cannot read frame {frame_num} from {video_path}')
    return _PIL_Image.fromarray(_cv2.cvtColor(bgr, _cv2.COLOR_BGR2RGB))

def _compute_and_cache_vgg16(img_data, save_path):
    convnet = _get_convnet()
    arr = _vgg16_mod.preprocess_input(_img_to_array(img_data))
    feat = convnet.predict(np.expand_dims(arr, axis=0), verbose=0)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as fid:
        pickle.dump(feat, fid, protocol=2)
    return feat

def _safe_load_images(self, img_sequences, bbox_sequences, ped_ids, save_path,
                      data_type='train', crop_type='none', crop_mode='warp',
                      crop_resize_ratio=2, regen_data=False):
    import pickle as _pkl
    from sf_gru import img_pad, jitter_bbox, squarify
    bbox_seq = bbox_sequences.copy()
    sequences = []
    for i, (seq, pid) in enumerate(zip(img_sequences, ped_ids)):
        img_seq = []
        for imp, b, p in zip(seq, bbox_seq[i], pid):
            flip_image = False
            set_id   = imp.split('/')[-3]
            vid_id   = imp.split('/')[-2]
            img_name = imp.split('/')[-1].split('.')[0]
            img_name_noflip = img_name.replace('_flip', '') if 'flip' in img_name else img_name
            if 'flip' in img_name:
                flip_image = True

            if crop_type == 'none':
                img_save_path = os.path.join(save_path, set_id, vid_id, img_name + '.pkl')
                noflip_path   = os.path.join(save_path, set_id, vid_id, img_name_noflip + '.pkl')
            else:
                img_save_path = os.path.join(save_path, set_id, vid_id, img_name + '_' + p[0] + '.pkl')
                noflip_path   = os.path.join(save_path, set_id, vid_id, img_name_noflip + '_' + p[0] + '.pkl')

            load_path = None
            if os.path.exists(img_save_path) and not regen_data:
                load_path = img_save_path
                flip_image = False
            elif flip_image and os.path.exists(noflip_path) and not regen_data:
                load_path = noflip_path

            feat = None
            if load_path:
                try:
                    with open(load_path, 'rb') as fid:
                        try:    feat = _pkl.load(fid)
                        except: feat = _pkl.load(fid, encoding='bytes')
                    if flip_image:
                        feat = feat[:, :, ::-1, :]
                except Exception:
                    log.warning('Corrupt cache file, regenerating: %s', load_path)
                    try: os.remove(load_path)
                    except OSError: pass
                    feat = None
            if feat is None:
                frame_num = int(img_name_noflip)
                img_data = _frame_from_video(set_id, vid_id, frame_num)
                if flip_image:
                    img_data = img_data.transpose(_PIL_Image.FLIP_LEFT_RIGHT)
                if crop_type == 'none':
                    img_data = img_data.resize((224, 224))
                elif crop_type == 'bbox':
                    img_data = img_pad(img_data.crop(list(map(int, b[0:4]))),
                                       mode=crop_mode, size=224)
                elif 'context' in crop_type:
                    bbox = jitter_bbox(imp, [b], 'enlarge', crop_resize_ratio)[0]
                    bbox = squarify(bbox, 1, img_data.size[0])
                    img_data = img_pad(img_data.crop(list(map(int, bbox[0:4]))),
                                       mode='pad_resize', size=224)
                elif 'surround' in crop_type:
                    from PIL import ImageDraw as _ID
                    b_org = list(map(int, b[0:4]))
                    bbox = jitter_bbox(imp, [b], 'enlarge', crop_resize_ratio)[0]
                    bbox = squarify(bbox, 1, img_data.size[0])
                    draw = _ID.Draw(img_data)
                    draw.rectangle(b_org, fill=(128, 128, 128))
                    del draw
                    img_data = img_pad(img_data.crop(list(map(int, bbox[0:4]))),
                                       mode='pad_resize', size=224)
                feat = _compute_and_cache_vgg16(img_data, noflip_path)

            if self._global_pooling == 'max':
                feat = np.squeeze(feat); feat = np.amax(feat, axis=0); feat = np.amax(feat, axis=0)
            elif self._global_pooling == 'avg':
                feat = np.squeeze(feat); feat = np.average(feat, axis=0); feat = np.average(feat, axis=0)
            else:
                feat = feat.ravel()
            img_seq.append(feat)
        sequences.append(img_seq)
    return np.array(sequences)

_sfgru_mod.SFGRU.load_images_crop_and_process = _safe_load_images

# ── Also patch get_pose for graceful zero-fallback on missing sets/frames ──────
_orig_get_pose = _sfgru_mod.SFGRU.get_pose

def _safe_get_pose(self, img_sequences, ped_ids, file_path, data_type='train'):
    import re
    preferred_backend = os.environ.get('PIE_POSE_BACKEND', 'rtmpose').strip().lower()
    set_poses_list = os.listdir(file_path)
    pose_files_by_set = {}
    for s in sorted(set_poses_list):
        match = re.match(r'^pose_(set\d+)(?:_(.+))?\.pkl$', s)
        if not match: continue
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
            try:    p = pickle.load(fid)
            except: p = pickle.load(fid, encoding='bytes')
        set_poses[set_id] = p
    log.info('Pose sources: %s', {k: v[1] for k, v in
             {s: next(((b,f) for b,f in pose_files_by_set[s] if b==preferred_backend),
                      pose_files_by_set[s][0]) for s in pose_files_by_set}.items()})

    poses_all = []
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
            k = img_name + '_' + p[0]
            if set_id not in set_poses or vid_id not in set_poses[set_id] or \
               k not in set_poses[set_id][vid_id]:
                pose.append([0] * 36)
            else:
                kp = set_poses[set_id][vid_id][k]
                pose.append(self.flip_pose(kp) if flip_image else kp)
        poses_all.append(pose)
    return np.array(poses_all)

_sfgru_mod.SFGRU.get_pose = _safe_get_pose

# ── Patched get_path so features land in PIE_data ─────────────────────────────
_FEAT_DIR = os.path.join(PIE_DATA_DIR, 'features')
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

from pie_data import PIE
from sf_gru import SFGRU
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             precision_score, recall_score)

# ── Standard PIE split opts (no set03-only restriction) ───────────────────────
DATA_OPTS = {
    'fstride': 1,
    'subset': 'default',
    'data_split_type': 'default',   # train=set01+02+04, val=set05+06, test=set03
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


def compute_metrics(labels, probs, threshold=0.5):
    binary = (probs >= threshold).astype(int)
    return {
        'acc':  accuracy_score(labels, binary),
        'f1':   f1_score(labels, binary, zero_division=0),
        'prec': precision_score(labels, binary, zero_division=0),
        'rec':  recall_score(labels, binary, zero_division=0),
        'auc':  roc_auc_score(labels, probs),
    }


def run_one_seed(seed, pose_backend, model_save_name):
    """Train and evaluate with a fixed random seed. Returns metric dict."""
    import tensorflow as tf
    tf.random.set_seed(seed)
    np.random.seed(seed)

    os.environ['PIE_POSE_BACKEND'] = pose_backend
    imdb = PIE(data_path=PIE_DATA_DIR)

    log.info('=== seed=%d  backend=%s ===', seed, pose_backend)

    beh_train = imdb.generate_data_trajectory_sequence('train', **DATA_OPTS)
    method = SFGRU()

    model_dir = os.path.join(SFGRU_DIR, 'data', 'models', 'pie',
                             f'{model_save_name}_seed{seed}')
    os.makedirs(model_dir, exist_ok=True)

    saved_files_path = method.train(beh_train, model_opts=MODEL_OPTS)

    import shutil
    default_dir = os.path.join(SFGRU_DIR, 'data', 'models', 'pie', 'sf-rnn')
    for f in os.listdir(default_dir):
        shutil.copy2(os.path.join(default_dir, f), model_dir)

    beh_test = imdb.generate_data_trajectory_sequence('test', **DATA_OPTS)
    acc, auc, f1, prec, rec = method.test(beh_test, saved_files_path)
    metrics = {'acc': acc, 'auc': auc, 'f1': f1, 'prec': prec, 'rec': rec,
               'seed': seed, 'backend': pose_backend}
    log.info('seed=%d  Acc=%.4f  AUC=%.4f  F1=%.4f', seed, acc, auc, f1)
    return metrics


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=5,
                        help='Number of independent training runs (default 5)')
    parser.add_argument('--backend', default='rtmpose',
                        choices=['rtmpose', 'openpose', 'none'],
                        help='Pose backend (default: rtmpose)')
    parser.add_argument('--no-pose-baseline', action='store_true',
                        help='Also run a zero-pose baseline for ablation')
    args = parser.parse_args()

    all_results = {}

    # ── Main experiment ──────────────────────────────────────────────────────
    runs = []
    for seed in range(args.seeds):
        m = run_one_seed(seed, args.backend, f'sf-rnn-{args.backend}-full')
        runs.append(m)
    all_results[args.backend] = {'runs': runs, 'summary': summarise(runs)}
    s = all_results[args.backend]['summary']
    log.info('\n%s (n=%d):  Acc %.4f±%.4f  AUC %.4f±%.4f  F1 %.4f±%.4f',
             args.backend, s['n'],
             s['acc_mean'], s['acc_std'],
             s['auc_mean'], s['auc_std'],
             s['f1_mean'],  s['f1_std'])

    # ── No-pose ablation (optional) ──────────────────────────────────────────
    if args.no_pose_baseline:
        orig_get_pose = _sfgru_mod.SFGRU.get_pose
        def _zero_pose(self, img_sequences, ped_ids, file_path, data_type='train'):
            poses = orig_get_pose(self, img_sequences, ped_ids, file_path, data_type)
            return np.zeros_like(poses)
        _sfgru_mod.SFGRU.get_pose = _zero_pose

        runs_nop = []
        for seed in range(args.seeds):
            m = run_one_seed(seed, 'none', 'sf-rnn-noPose-full')
            runs_nop.append(m)
        _sfgru_mod.SFGRU.get_pose = orig_get_pose
        all_results['no_pose'] = {'runs': runs_nop, 'summary': summarise(runs_nop)}
        s2 = all_results['no_pose']['summary']
        log.info('\nno_pose (n=%d):  Acc %.4f±%.4f  AUC %.4f±%.4f  F1 %.4f±%.4f',
                 s2['n'], s2['acc_mean'], s2['acc_std'],
                 s2['auc_mean'], s2['auc_std'], s2['f1_mean'], s2['f1_std'])

    # ── Save ──────────────────────────────────────────────────────────────────
    out = os.path.join(RESULTS_DIR, 'full_pie_results.pkl')
    with open(out, 'wb') as f:
        pickle.dump(all_results, f)
    log.info('Saved: %s', out)

    # ── Print table ───────────────────────────────────────────────────────────
    print('\n' + '='*62)
    print(f'{"Backend":<14} {"Acc":>8} {"±":>5} {"AUC":>8} {"±":>5} {"F1":>8} {"±":>5}')
    print('-'*62)
    for name, res in all_results.items():
        s = res['summary']
        print(f'{name:<14} {s["acc_mean"]:>8.4f} {s["acc_std"]:>5.4f} '
              f'{s["auc_mean"]:>8.4f} {s["auc_std"]:>5.4f} '
              f'{s["f1_mean"]:>8.4f} {s["f1_std"]:>5.4f}')
    print('='*62)


if __name__ == '__main__':
    main()
