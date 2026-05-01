"""
train_rtmpose.py
================
Retrains SF-GRU using RTMPose poses (pose_set03_rtmpose.pkl) and evaluates
against the OpenPose baseline. Saves results to results/rtmpose_comparison.pkl.

Usage:
  conda run -n sfgru_eval python train_rtmpose.py
"""

import os, sys, pickle, types, logging
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
SFGRU_DIR    = '/media/emma/data/MMML/SF-GRU'
PIE_UTIL_DIR = '/media/emma/data/MMML/PIE/utilities'
PIE_DATA_DIR = '/media/emma/data/MMML/PIE_data'
RESULTS_DIR  = os.path.join(SFGRU_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

sys.path.insert(0, SFGRU_DIR)
sys.path.append(PIE_UTIL_DIR)
os.chdir(SFGRU_DIR)

# ── Keras compatibility patches ───────────────────────────────────────────────
import keras
import keras.layers as kl
import keras.regularizers as kr

_rec = types.ModuleType('keras.layers.recurrent')
_rec.GRU = kl.GRU
sys.modules['keras.layers.recurrent'] = _rec

_core = types.ModuleType('keras.layers.core')
_core.regularizers = kr
sys.modules['keras.layers.core'] = _core

import utils as _sfgru_utils
import sf_gru as _sfgru_mod

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

_sfgru_utils.get_path = _patched_get_path
_sfgru_mod.get_path   = _patched_get_path

from pie_data import PIE
from sf_gru import SFGRU
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

# ── Patch get_pose to return zeros for sets without a pose file ───────────────
import sf_gru as _sfgru_mod
_original_get_pose = _sfgru_mod.SFGRU.get_pose

def _safe_get_pose(self, img_sequences, ped_ids, file_path, data_type='train'):
    import re, pickle, numpy as np
    # Build set_poses normally
    preferred_backend = os.environ.get('PIE_POSE_BACKEND', 'openpose').strip().lower()
    set_poses_list = os.listdir(file_path)
    set_poses = {}
    pose_files_by_set = {}
    for s in sorted(set_poses_list):
        pose_meta = self._parse_pose_cache_name(s)
        if pose_meta is None: continue
        set_id, backend = pose_meta
        pose_files_by_set.setdefault(set_id, []).append((backend, s))
    for set_id, candidates in pose_files_by_set.items():
        chosen = next(((b,f) for b,f in candidates if b==preferred_backend), None)
        if chosen is None:
            chosen = next(((b,f) for b,f in candidates if b=='openpose'), None)
        if chosen is None:
            chosen = candidates[0]
        backend, file_name = chosen
        with open(os.path.join(file_path, file_name), 'rb') as fid:
            try: p = pickle.load(fid)
            except: p = pickle.load(fid, encoding='bytes')
        set_poses[set_id] = p
    print('Pose sources loaded for sets:', list(set_poses.keys()))

    poses_all = []
    for seq, pid in zip(img_sequences, ped_ids):
        pose = []
        for imp, p in zip(seq, pid):
            flip_image = False
            set_id  = imp.split('/')[-3]
            vid_id  = imp.split('/')[-2]
            img_name = imp.split('/')[-1].split('.')[0]
            if 'flip' in img_name:
                img_name = img_name.replace('_flip', '')
                flip_image = True
            k = img_name + '_' + p[0]
            # Gracefully return zeros for sets without pose data
            if set_id not in set_poses or vid_id not in set_poses[set_id] or k not in set_poses[set_id][vid_id]:
                pose.append([0] * 36)
            else:
                if flip_image:
                    pose.append(self.flip_pose(set_poses[set_id][vid_id][k]))
                else:
                    pose.append(set_poses[set_id][vid_id][k])
        poses_all.append(pose)
    return np.array(poses_all)

_sfgru_mod.SFGRU.get_pose = _safe_get_pose

# ── Restrict PIE to set03 only ────────────────────────────────────────────────
from pie_data import PIE as _PIE_cls
_orig_get_image_set_ids = _PIE_cls._get_image_set_ids

def _set03_only(self, image_set):
    # Always return set03 regardless of train/test/val/all
    return ['set03']

_PIE_cls._get_image_set_ids = _set03_only

# ── Patch load_images_crop_and_process to use non-flip cache for flipped seqs ─
import numpy as _np
_orig_load = _sfgru_mod.SFGRU.load_images_crop_and_process

def _safe_load_images(self, img_sequences, bbox_sequences, ped_ids, save_path,
                      data_type='train', crop_type='none', crop_mode='warp',
                      crop_resize_ratio=2, regen_data=False):
    import pickle as _pkl
    bbox_seq = bbox_sequences.copy()
    sequences = []
    for i, (seq, pid) in enumerate(zip(img_sequences, ped_ids)):
        img_seq = []
        for imp, b, p in zip(seq, bbox_seq[i], pid):
            flip_image = False
            set_id   = imp.split('/')[-3]
            vid_id   = imp.split('/')[-2]
            img_name = imp.split('/')[-1].split('.')[0]

            if 'flip' in img_name:
                flip_image = True
                img_name_noflip = img_name.replace('_flip', '')
            else:
                img_name_noflip = img_name

            if crop_type == 'none':
                img_save_path = os.path.join(save_path, set_id, vid_id, img_name + '.pkl')
                noflip_path   = os.path.join(save_path, set_id, vid_id, img_name_noflip + '.pkl')
            else:
                img_save_path = os.path.join(save_path, set_id, vid_id, img_name + '_' + p[0] + '.pkl')
                noflip_path   = os.path.join(save_path, set_id, vid_id, img_name_noflip + '_' + p[0] + '.pkl')

            # Try flip cache first, then non-flip cache (apply flip in memory)
            load_path = None
            if os.path.exists(img_save_path) and not regen_data:
                load_path = img_save_path
                flip_image = False  # already flipped if stored
            elif flip_image and os.path.exists(noflip_path) and not regen_data:
                load_path = noflip_path  # load non-flip, flip below

            if load_path:
                with open(load_path, 'rb') as fid:
                    try:    feat = _pkl.load(fid)
                    except: feat = _pkl.load(fid, encoding='bytes')
                if flip_image:
                    feat = feat[:, :, ::-1, :]  # horizontal flip of (1, H, W, C)
                # Apply same pooling as original load_images_crop_and_process
                if self._global_pooling == 'max':
                    feat = _np.squeeze(feat)
                    feat = _np.amax(feat, axis=0)
                    feat = _np.amax(feat, axis=0)
                elif self._global_pooling == 'avg':
                    feat = _np.squeeze(feat)
                    feat = _np.average(feat, axis=0)
                    feat = _np.average(feat, axis=0)
                else:
                    feat = feat.ravel()
                img_seq.append(feat)
            else:
                # Fall back to original method (needs raw image)
                # This should only happen for truly uncached non-flip frames
                from keras.preprocessing.image import load_img, img_to_array
                actual_imp = imp.replace('_flip', '') if flip_image else imp
                from PIL import Image
                img_data = load_img(actual_imp)
                if flip_image:
                    img_data = img_data.transpose(Image.FLIP_LEFT_RIGHT)
                from sf_gru import img_pad, jitter_bbox, squarify
                if crop_type == 'bbox':
                    cropped = img_data.crop(list(map(int, b[0:4])))
                    img_data = img_pad(cropped, mode=crop_mode, size=224)
                feat = img_to_array(img_data) / 255.0
                os.makedirs(os.path.dirname(img_save_path), exist_ok=True)
                with open(img_save_path, 'wb') as fid:
                    _pkl.dump(feat, fid)
                img_seq.append(feat)
        sequences.append(img_seq)
    return _np.array(sequences)

_sfgru_mod.SFGRU.load_images_crop_and_process = _safe_load_images

# ── Config ────────────────────────────────────────────────────────────────────
# Use 'random' split within set03 only.
# We patch _get_image_set_ids to always return set03 so training never
# tries to load images or poses from set01/02/04/05/06.
DATA_OPTS = {
    'fstride': 1, 'subset': 'default', 'data_split_type': 'random',
    'seq_type': 'crossing', 'min_track_size': 75,
}
MODEL_OPTS = {
    'obs_input_type': ['local_box', 'local_context', 'pose', 'box', 'speed'],
    'enlarge_ratio': 1.5, 'pred_target_type': ['crossing'],
    'obs_length': 15, 'time_to_event': 60,
    'dataset': 'pie', 'normalize_boxes': True,
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

def run_experiment(pose_backend, model_save_name):
    print('\n' + '='*60)
    print('EXPERIMENT: pose_backend=%s' % pose_backend)
    print('='*60)
    os.environ['PIE_POSE_BACKEND'] = pose_backend

    imdb = PIE(data_path=PIE_DATA_DIR)

    # Train
    beh_train = imdb.generate_data_trajectory_sequence('train', **DATA_OPTS)
    method = SFGRU()

    # Save model to a unique path so experiments don't overwrite each other
    orig_model_path = os.path.join(SFGRU_DIR, 'data', 'models', 'pie', 'sf-rnn')
    exp_model_path  = os.path.join(SFGRU_DIR, 'data', 'models', 'pie', model_save_name)
    os.makedirs(exp_model_path, exist_ok=True)

    saved_files_path = method.train(beh_train, model_opts=MODEL_OPTS)

    # Move trained model to experiment path
    import shutil
    for f in os.listdir(orig_model_path):
        shutil.copy2(os.path.join(orig_model_path, f), exp_model_path)
    print('Model saved to:', exp_model_path)

    # Test
    beh_test = imdb.generate_data_trajectory_sequence('test', **DATA_OPTS)
    acc, auc, f1, prec, rec = method.test(beh_test, saved_files_path)

    results = {'acc': acc, 'auc': auc, 'f1': f1, 'prec': prec, 'rec': rec,
               'pose_backend': pose_backend}
    print('\nResults: Acc=%.4f  AUC=%.4f  F1=%.4f' % (acc, auc, f1))
    return results

# ── Run both experiments ──────────────────────────────────────────────────────
results = {}

# 1. RTMPose
results['rtmpose'] = run_experiment('rtmpose', 'sf-rnn-rtmpose')

# 2. OpenPose baseline — retrain from scratch with same split for fair comparison
results['openpose'] = run_experiment('openpose', 'sf-rnn-openpose')

# ── Print comparison ──────────────────────────────────────────────────────────
print('\n' + '='*60)
print('FINAL RESULTS')
print('='*60)
print('%-12s %8s %8s %8s %8s %8s' % ('Backend', 'Acc', 'AUC', 'F1', 'Prec', 'Rec'))
print('-'*56)
for name, m in results.items():
    print('%-12s %8.4f %8.4f %8.4f %8.4f %8.4f' % (
        name, m['acc'], m['auc'], m['f1'], m['prec'], m['rec']))

# Save
out_path = os.path.join(RESULTS_DIR, 'rtmpose_comparison.pkl')
with open(out_path, 'wb') as f:
    pickle.dump(results, f)
print('\nSaved to:', out_path)
