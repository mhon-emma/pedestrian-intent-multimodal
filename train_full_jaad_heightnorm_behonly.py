"""
train_full_jaad_heightnorm.py
===============================
Same as train_full_jaad_nospeed.py, but uses HeightNormSFGRUTorch
(sf_gru_torch_heightnorm.py) instead of SFGRUTorch -- see
train_full_pie_heightnorm.py and sf_gru_torch_heightnorm.py for the full
motivation and correctness notes.

Usage
-----
  python train_full_jaad_heightnorm.py --seeds 3 --backend rtmpose

Output
------
  results/full_jaad_heightnorm_results_<backend>.pkl
"""

import argparse
import logging
import os
import pickle
import sys

import numpy as np
import torch
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

SFGRU_DIR    = '/usr1/home/mehon/emma_pedestrian-intent-multimodal'
JAAD_UTIL_DIR = '/usr1/home/mehon/JAAD'
JAAD_DATA_DIR = '/usr1/home/mehon/JAAD'
RESULTS_DIR   = os.path.join(SFGRU_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

sys.path.insert(0, SFGRU_DIR)
sys.path.append(JAAD_UTIL_DIR)
os.chdir(SFGRU_DIR)

import sf_gru_torch as _sfgru_mod
from jaad_data import JAAD
from sf_gru_torch import img_pad, squarify
from sf_gru_torch_heightnorm import HeightNormSFGRUTorch

# -- JAAD-specific image loader (identical to train_full_jaad_nospeed.py's) ----
def _load_images_crop_and_process_jaad(self, img_sequences, bbox_sequences,
                                       ped_ids, save_path,
                                       data_type='train', crop_type='none',
                                       crop_mode='warp', crop_resize_ratio=2,
                                       regen_data=False):
    from utils import jitter_bbox
    sequences = []
    bbox_seq = bbox_sequences.copy()
    for i, (seq, pid) in enumerate(zip(img_sequences, ped_ids)):
        img_seq = []
        for imp, b, p in zip(seq, bbox_seq[i], pid):
            flip_image = False
            vid_id   = imp.split('/')[-2]
            img_name = imp.split('/')[-1].split('.')[0]

            img_save_folder = os.path.join(save_path, vid_id)
            if crop_type == 'none':
                img_save_path = os.path.join(img_save_folder, img_name + '.pkl')
            else:
                img_save_path = os.path.join(img_save_folder, img_name + '_' + p[0] + '.pkl')

            cached = None
            if os.path.exists(img_save_path) and not regen_data:
                try:
                    with open(img_save_path, 'rb') as fid:
                        cached = pickle.load(fid)
                except (pickle.UnpicklingError, EOFError):
                    cached = None
            if cached is not None:
                img_features = cached
            else:
                if 'flip' in imp:
                    imp = imp.replace('_flip', '')
                    flip_image = True
                if crop_type == 'none':
                    img_data = Image.open(imp).convert('RGB').resize((224, 224))
                    if flip_image:
                        img_data = img_data.transpose(Image.FLIP_LEFT_RIGHT)
                else:
                    img_data = Image.open(imp).convert('RGB')
                    if flip_image:
                        img_data = img_data.transpose(Image.FLIP_LEFT_RIGHT)
                    if crop_type == 'bbox':
                        cropped_image = img_data.crop(list(map(int, b[0:4])))
                        img_data = img_pad(cropped_image, mode=crop_mode, size=224)
                    elif 'context' in crop_type:
                        bbox = jitter_bbox(imp, [b], 'enlarge', crop_resize_ratio)[0]
                        bbox = squarify(bbox, 1, img_data.size[0])
                        bbox = list(map(int, bbox[0:4]))
                        cropped_image = img_data.crop(bbox)
                        img_data = img_pad(cropped_image, mode='pad_resize', size=224)
                    elif 'surround' in crop_type:
                        from PIL import ImageDraw
                        b_org = [b[0], b[1], b[2], b[3]]
                        bbox = jitter_bbox(imp, [b], 'enlarge', crop_resize_ratio)[0]
                        bbox = squarify(bbox, 1, img_data.size[0])
                        bbox = list(map(int, bbox[0:4]))
                        draw = ImageDraw.Draw(img_data)
                        draw.rectangle(b_org, fill=(128, 128, 128))
                        del draw
                        cropped_image = img_data.crop(bbox)
                        img_data = img_pad(cropped_image, mode='pad_resize', size=224)
                    else:
                        raise ValueError('ERROR: Undefined value for crop_type {}!'.format(crop_type))
                img_features = self._vgg_forward(img_data)
                if not os.path.exists(img_save_folder):
                    os.makedirs(img_save_folder, exist_ok=True)
                tmp_path = '%s.tmp.%d' % (img_save_path, os.getpid())
                with open(tmp_path, 'wb') as fid:
                    pickle.dump(img_features, fid, pickle.HIGHEST_PROTOCOL)
                os.replace(tmp_path, img_save_path)

            if self._global_pooling == 'max':
                img_features = np.squeeze(img_features)
                img_features = np.amax(img_features, axis=0)
                img_features = np.amax(img_features, axis=0)
            elif self._global_pooling == 'avg':
                img_features = np.squeeze(img_features)
                img_features = np.average(img_features, axis=0)
                img_features = np.average(img_features, axis=0)
            else:
                img_features = img_features.ravel()

            img_seq.append(img_features)
        sequences.append(img_seq)
    return np.array(sequences)

_sfgru_mod.SFGRUTorch.load_images_crop_and_process = _load_images_crop_and_process_jaad

# -- get_pose for JAAD: single pkl keyed by {vid_id: {key: [36 floats]}} -------
_JAAD_POSE_FILES = {
    'rtmpose': 'pose_jaad_rtmpose.pkl',
    'openpose': 'pose_jaad_openpose_full.pkl',
}

def _safe_get_pose_jaad(self, img_sequences, ped_ids, file_path, data_type='train'):
    backend = os.environ.get('JAAD_POSE_BACKEND', 'rtmpose').strip().lower()
    pose_file = _JAAD_POSE_FILES.get(backend, _JAAD_POSE_FILES['rtmpose'])
    pose_pkl = os.path.join(file_path, pose_file)
    with open(pose_pkl, 'rb') as fid:
        vid_poses = pickle.load(fid)
    log.info('Loaded JAAD %s poses (%s) for %d videos', backend, pose_file, len(vid_poses))

    poses_all = []
    for seq, pid in zip(img_sequences, ped_ids):
        pose = []
        for imp, p in zip(seq, pid):
            flip_image = False
            vid_id   = imp.split('/')[-2]
            img_name = imp.split('/')[-1].split('.')[0]
            if 'flip' in img_name:
                img_name = img_name.replace('_flip', '')
                flip_image = True
            k = img_name + '_' + p[0]
            if vid_id not in vid_poses or k not in vid_poses[vid_id]:
                pose.append([0] * 36)
            else:
                kp = vid_poses[vid_id][k]
                pose.append(self.flip_pose(kp) if flip_image else kp)
        poses_all.append(pose)
    return np.array(poses_all)

_sfgru_mod.SFGRUTorch.get_pose = _safe_get_pose_jaad

# -- Patched get_path so features/models land under this repo's data/ tree -----
_FEAT_DIR = os.path.join(SFGRU_DIR, 'data', 'features')
_POSE_DIR = os.path.join(SFGRU_DIR, 'data', 'features', 'jaad', 'poses')

def _patched_get_path(file_name='', save_folder='models', dataset='jaad',
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

from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

DATA_OPTS = {
    'sample_type': 'beh',  # CRITICAL FIX: JAAD's default 'all' silently includes
                           # non-behavior-annotated pedestrians (~75% of every split)
                           # whose crossing label is hardcoded to 0 in jaad_data.py's
                           # _get_crossing() -- not because they were observed not
                           # crossing, but because they were never behaviorally
                           # annotated. See train_full_jaad_nospeed_behonly.py for the
                           # full investigation. PIE has no equivalent category and is
                           # unaffected.
    'fstride': 1,
    'subset': 'default',
    'data_split_type': 'default',   # uses JAAD/split_ids/default/{train,val,test}.txt
    'seq_type': 'crossing',
    'height_rng': [0, float('inf')],
    'squarify_ratio': 0,
    'min_track_size': 75,
    'random_params': {'ratios': None, 'val_data': True, 'regen_data': False},
    'kfold_params': {'num_folds': 5, 'fold': 1},
}
MODEL_OPTS = {
    'obs_input_type': ['local_box', 'local_context', 'pose', 'box'],  # no speed
    'enlarge_ratio': 1.5,
    'pred_target_type': ['crossing'],
    'obs_length': 15,
    'time_to_event': 60,
    'dataset': 'jaad_heightnorm_behonly',  # distinct cache from jaad_nospeed
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


def run_one_seed(seed, pose_backend, model_save_name, experiment_label=None):
    if experiment_label is None:
        experiment_label = pose_backend
    torch.manual_seed(seed)
    np.random.seed(seed)

    os.environ['JAAD_POSE_BACKEND'] = pose_backend
    imdb = JAAD(data_path=JAAD_DATA_DIR)

    log.info('=== seed=%d  experiment=%s (pose_cache=%s) ===', seed, experiment_label, pose_backend)

    beh_train = imdb.generate_data_trajectory_sequence('train', **DATA_OPTS)
    beh_val   = imdb.generate_data_trajectory_sequence('val', **DATA_OPTS)
    method = HeightNormSFGRUTorch()

    saved_files_path = method.train(beh_train, data_val=beh_val, model_opts=MODEL_OPTS)

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
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--backend', default='rtmpose', choices=['rtmpose', 'openpose', 'none'])
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
        m = run_one_seed(seed, pose_env_backend, f'sf-rnn-{args.backend}-jaad-heightnorm',
                         experiment_label=args.backend)
        runs.append(m)

    summary = summarise(runs)
    log.info('\n%s (n=%d):  Acc %.4f+/-%.4f  AUC %.4f+/-%.4f  F1 %.4f+/-%.4f',
             args.backend, summary['n'],
             summary['acc_mean'], summary['acc_std'],
             summary['auc_mean'], summary['auc_std'],
             summary['f1_mean'],  summary['f1_std'])

    out = os.path.join(RESULTS_DIR, f'full_jaad_heightnorm_behonly_results_{args.backend}.pkl')
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
