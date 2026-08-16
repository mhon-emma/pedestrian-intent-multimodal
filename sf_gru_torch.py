"""
sf_gru_torch.py
================
PyTorch port of SFGRU (sf_gru.py) — same stacked-GRU fusion architecture and
data pipeline (image cropping, VGG16 feature caching, pose/box/speed sequence
generation), reimplemented without a Keras/TensorFlow dependency.

Data-prep logic (get_data_sequence, get_data_sequence_balance, get_pose,
flip_pose, get_data, load_images_crop_and_process) is carried over from
sf_gru.py essentially unchanged — it was already pure NumPy/PIL. The model,
training loop, and VGG16 feature extractor are new: VGG16 now uses
torchvision's ImageNet weights instead of Keras's, so any previously cached
Keras-VGG16 .pkl features are not reused (different weights/preprocessing) —
caches regenerate on first use into the same data/features/<dataset>/... tree.

Why this exists: sf_gru.py's SFGRU imports Keras at module level, so it can't
be reused as a library on a TF-broken environment. This module has no Keras
import anywhere.
"""

import os
import pickle
import re
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from torchvision import models as tv_models
from torchvision import transforms as tv_transforms

from utils import (bbox_sanity_check, get_path, img_pad, jitter_bbox,
                   squarify, update_progress)

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


class SFGRUTorch(object):
    """
    PyTorch reimplementation of SFGRU. See sf_gru.py for the original
    Keras/TF version and paper reference.
    """

    def __init__(self,
                num_hidden_units=256,
                global_pooling='avg',
                regularizer_val=0.0001,
                device=None):
        self._num_hidden_units = num_hidden_units
        self._regularizer_value = regularizer_val
        self._global_pooling = global_pooling
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self._vgg = None
        self._vgg_transform = tv_transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)

    # -- VGG16 feature extractor (torchvision, ImageNet weights) --------------
    def _get_vgg(self):
        if self._vgg is None:
            vgg = tv_models.vgg16(weights='IMAGENET1K_V1')
            self._vgg = vgg.features.to(self.device).eval()
            for p in self._vgg.parameters():
                p.requires_grad = False
        return self._vgg

    @torch.no_grad()
    def _vgg_forward(self, img_data):
        """img_data: PIL Image, 224x224. Returns (1, 7, 7, 512) numpy array
        to match the original Keras VGG16 (channels-last) cache format."""
        arr = np.asarray(img_data, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
        t = self._vgg_transform(t)
        feat = self._get_vgg()(t)  # (1, 512, 7, 7)
        feat = feat.permute(0, 2, 3, 1).cpu().numpy()  # -> (1, 7, 7, 512)
        return feat

    # -- Image loading / cropping / feature caching ----------------------------
    def load_images_crop_and_process(self, img_sequences, bbox_sequences,
                                     ped_ids, save_path,
                                     data_type='train',
                                     crop_type='none',
                                     crop_mode='warp',
                                     crop_resize_ratio=2,
                                     regen_data=False):
        print("Generating {} features crop_type={} crop_mode={}\
              \nsave_path={}, ".format(data_type, crop_type, crop_mode,
              save_path))
        sequences = []
        bbox_seq = bbox_sequences.copy()
        i = -1
        for seq, pid in zip(img_sequences, ped_ids):
            i += 1
            update_progress(i / len(img_sequences))
            img_seq = []
            for imp, b, p in zip(seq, bbox_seq[i], pid):
                flip_image = False
                set_id = imp.split('/')[-3]
                vid_id = imp.split('/')[-2]
                img_name = imp.split('/')[-1].split('.')[0]

                img_save_folder = os.path.join(save_path, set_id, vid_id)
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
                        # Another concurrent run's write was caught mid-flight;
                        # treat as a cache miss and recompute.
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
                            b_org = [b[0], b[1], b[2], b[3]]
                            bbox = jitter_bbox(imp, [b], 'enlarge', crop_resize_ratio)[0]
                            bbox = squarify(bbox, 1, img_data.size[0])
                            bbox = list(map(int, bbox[0:4]))
                            draw = ImageDraw.Draw(img_data)
                            draw.rectangle([b_org[0], b_org[1], b_org[2], b_org[3]],
                                           fill=(128, 128, 128))
                            del draw
                            cropped_image = img_data.crop(bbox)
                            img_data = img_pad(cropped_image, mode='pad_resize', size=224)
                        else:
                            raise ValueError('ERROR: Undefined value for crop_type {}!'.format(crop_type))
                    img_features = self._vgg_forward(img_data)
                    if not os.path.exists(img_save_folder):
                        os.makedirs(img_save_folder, exist_ok=True)
                    # Write to a per-process temp file and atomically rename into
                    # place, so a concurrent reader (e.g. a second experiment run
                    # sharing this cache dir) never observes a partially-written
                    # pickle -- it either sees the old file or the complete new one.
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
        sequences = np.array(sequences)
        return sequences

    # -- Pose loading (unchanged from sf_gru.py) --------------------------------
    def _parse_pose_cache_name(self, file_name):
        match = re.match(r'^pose_(set\d+)(?:_(.+))?\.pkl$', file_name)
        if not match:
            return None
        set_id = match.group(1)
        backend = (match.group(2) or 'openpose').lower()
        return set_id, backend

    def get_pose(self, img_sequences, ped_ids, file_path, data_type='train'):
        print('\n#####################################')
        print('Getting poses %s' % data_type)
        print('#####################################')
        poses_all = []
        preferred_backend = os.environ.get('PIE_POSE_BACKEND', 'yolo').strip().lower() or 'yolo'
        set_poses_list = os.listdir(file_path)
        set_poses = {}
        pose_sources = {}
        pose_files_by_set = {}
        for s in sorted(set_poses_list):
            pose_meta = self._parse_pose_cache_name(s)
            if pose_meta is None:
                continue
            set_id, backend = pose_meta
            pose_files_by_set.setdefault(set_id, []).append((backend, s))

        for set_id in sorted(pose_files_by_set.keys()):
            candidates = pose_files_by_set[set_id]
            chosen = None
            for backend, file_name in candidates:
                if backend == preferred_backend:
                    chosen = (backend, file_name)
                    break
            if chosen is None:
                for backend, file_name in candidates:
                    if backend == 'openpose':
                        chosen = (backend, file_name)
                        break
            if chosen is None:
                chosen = candidates[0]

            backend, file_name = chosen
            with open(os.path.join(file_path, file_name), 'rb') as fid:
                p = pickle.load(fid)
            set_poses[set_id] = p
            pose_sources[set_id] = '%s (%s)' % (file_name, backend)
        print(pose_sources)
        i = -1
        missing = 0
        total = 0
        for seq, pid in zip(img_sequences, ped_ids):
            i += 1
            update_progress(i / len(img_sequences))
            pose = []
            for imp, p in zip(seq, pid):
                flip_image = False
                set_id = imp.split('/')[-3]
                vid_id = imp.split('/')[-2]
                img_name = imp.split('/')[-1].split('.')[0]
                if 'flip' in img_name:
                    img_name = img_name.replace('_flip', '')
                    flip_image = True
                total += 1
                vid_poses = set_poses[set_id][vid_id]
                # OpenPose-style cache key: frame_<ped_id> (ped_id already
                # includes set/vid, e.g. '3_4_344' -> '01379_3_4_344').
                k = img_name + '_' + p[0]
                # RTMPose-style cache key (extract_rtmpose.py): an extra
                # set_num/vid_num segment is inserted before ped_id, e.g.
                # frame='01013', set_id='set01' -> set_num='1',
                # vid_id='video_0001' -> vid_num='0001', ped_id='1_1_1'
                # -> '01013_1_0001_1_1_1'.
                if k not in vid_poses:
                    set_num = set_id.replace('set', '').lstrip('0') or '0'
                    vid_num = vid_id.replace('video_', '')
                    k_rtmpose = '%s_%s_%s_%s' % (img_name, set_num, vid_num, p[0])
                    if k_rtmpose in vid_poses:
                        k = k_rtmpose
                if k in vid_poses:
                    if flip_image:
                        pose.append(self.flip_pose(vid_poses[k]))
                    else:
                        pose.append(vid_poses[k])
                else:
                    missing += 1
                    pose.append([0] * 36)
            poses_all.append(pose)
        poses_all = np.array(poses_all)
        if total:
            print('Pose lookup: %d/%d frames missing (%.1f%%)' %
                  (missing, total, 100.0 * missing / total))
        return poses_all

    def flip_pose(self, pose):
        flip_map = [0, 1, 2, 3, 10, 11, 12, 13, 14, 15, 4, 5, 6, 7, 8, 9, 22, 23, 24, 25,
                    26, 27, 16, 17, 18, 19, 20, 21, 30, 31, 28, 29, 34, 35, 32, 33]
        new_pose = pose.copy()
        flip_pose = [0] * len(new_pose)
        for i in range(len(new_pose)):
            if i % 2 == 0 and new_pose[i] != 0:
                new_pose[i] = 1 - new_pose[i]
            flip_pose[flip_map[i]] = new_pose[i]
        return flip_pose

    # -- Sequence generation (unchanged from sf_gru.py) -------------------------
    def get_data_sequence(self, data_raw, obs_length, time_to_event, normalize):
        print('\n#####################################')
        print('Generating raw data')
        print('#####################################')
        d = {'center': data_raw['center'].copy(),
             'box': data_raw['bbox'].copy(),
             'box_org': data_raw['bbox'].copy(),
             'ped_id': data_raw['pid'].copy(),
             'acts': data_raw['activities'].copy(),
             'image': data_raw['image'].copy()}

        try:
            d['speed'] = data_raw['obd_speed'].copy()
        except KeyError:
            d['speed'] = data_raw['vehicle_act'].copy()
            print('Jaad dataset does not have speed information')
            print('Vehicle actions are used instead')

        # Some dataset interfaces (observed on JAAD with min_track_size=0)
        # can return fully-empty samples (every field == []) for tracks that
        # should have been filtered out. Drop them before any per-sample
        # indexing, since an empty 'acts' entry has no [0] to read.
        keep = [i for i in range(len(d['acts'])) if len(d['acts'][i]) > 0]
        if len(keep) != len(d['acts']):
            print('Dropping %d empty sample(s) (indices %s)' %
                 (len(d['acts']) - len(keep), [i for i in range(len(d['acts'])) if i not in keep]))
            for k in d:
                d[k] = [d[k][i] for i in keep]

        for i in range(len(d['box'])):
            d['box'][i] = d['box'][i][- obs_length - time_to_event:-time_to_event]
            d['center'][i] = d['center'][i][- obs_length - time_to_event:-time_to_event]
            if normalize:
                d['box'][i] = np.subtract(d['box'][i][1:], d['box'][i][0]).tolist()
                d['center'][i] = np.subtract(d['center'][i][1:], d['center'][i][0]).tolist()

        if normalize:
            obs_length -= 1

        for k in d.keys():
            if k != 'box' and k != 'center':
                for i in range(len(d[k])):
                    d[k][i] = d[k][i][- obs_length - time_to_event:-time_to_event]
                d[k] = np.array(d[k])
            else:
                d[k] = np.array(d[k])
        d['acts'] = d['acts'][:, 0, :]
        return d

    def get_data_sequence_balance(self, data_raw, obs_length, time_to_event, normalize):
        print('\n#####################################')
        print('Generating balanced raw data')
        print('#####################################')
        d = {'center': data_raw['center'].copy(),
             'box': data_raw['bbox'].copy(),
             'ped_id': data_raw['pid'].copy(),
             'acts': data_raw['activities'].copy(),
             'image': data_raw['image'].copy()}

        try:
            d['speed'] = data_raw['obd_speed'].copy()
        except Exception:
            d['speed'] = data_raw['vehicle_act'].copy()
            print('Jaad dataset does not have speed information')
            print('Vehicle actions are used instead')

        # See get_data_sequence for why this guard is needed (observed on
        # JAAD with min_track_size=0: some fully-empty samples get through).
        keep = [i for i in range(len(d['acts'])) if len(d['acts'][i]) > 0]
        if len(keep) != len(d['acts']):
            print('Dropping %d empty sample(s) (indices %s)' %
                 (len(d['acts']) - len(keep), [i for i in range(len(d['acts'])) if i not in keep]))
            for k in d:
                d[k] = [d[k][i] for i in keep]

        gt_labels = [gt[0] for gt in d['acts']]
        num_pos_samples = np.count_nonzero(np.array(gt_labels))
        num_neg_samples = len(gt_labels) - num_pos_samples

        if num_neg_samples == num_pos_samples:
            print('Positive and negative samples are already balanced')
        else:
            print('Unbalanced: \t Positive: {} \t Negative: {}'.format(num_pos_samples, num_neg_samples))
            gt_augment = 1 if num_neg_samples > num_pos_samples else 0

            img_width = data_raw['image_dimension'][0]
            num_samples = len(d['ped_id'])
            for i in range(num_samples):
                if d['acts'][i][0][0] == gt_augment:
                    flipped = d['center'][i].copy()
                    flipped = [[img_width - c[0], c[1]] for c in flipped]
                    d['center'].append(flipped)
                    flipped = d['box'][i].copy()

                    flipped = [np.array([img_width - c[2], c[1], img_width - c[0], c[3]])
                               for c in flipped]
                    d['box'].append(flipped)

                    d['ped_id'].append(data_raw['pid'][i].copy())
                    d['acts'].append(d['acts'][i].copy())
                    flipped = d['image'][i].copy()
                    flipped = [c.replace('.png', '_flip.png') for c in flipped]

                    d['image'].append(flipped)
                    if 'speed' in d.keys():
                        d['speed'].append(d['speed'][i].copy())
            gt_labels = [gt[0] for gt in d['acts']]
            num_pos_samples = np.count_nonzero(np.array(gt_labels))
            num_neg_samples = len(gt_labels) - num_pos_samples
            if num_neg_samples > num_pos_samples:
                rm_index = np.where(np.array(gt_labels) == 0)[0]
            else:
                rm_index = np.where(np.array(gt_labels) == 1)[0]

            dif_samples = abs(num_neg_samples - num_pos_samples)
            np.random.seed(42)
            np.random.shuffle(rm_index)
            rm_index = rm_index[0:dif_samples]

            for k in d:
                seq_data_k = d[k]
                d[k] = [seq_data_k[i] for i in range(0, len(seq_data_k)) if i not in rm_index]

            new_gt_labels = [gt[0] for gt in d['acts']]
            num_pos_samples = np.count_nonzero(np.array(new_gt_labels))
            print('Balanced:\t Positive: %d  \t Negative: %d\n'
                  % (num_pos_samples, len(d['acts']) - num_pos_samples))

        d['box_org'] = d['box'].copy()

        for i in range(len(d['box'])):
            d['box'][i] = d['box'][i][- obs_length - time_to_event:-time_to_event]
            d['center'][i] = d['center'][i][- obs_length - time_to_event:-time_to_event]
            if normalize:
                d['box'][i] = np.subtract(d['box'][i][1:], d['box'][i][0]).tolist()
                d['center'][i] = np.subtract(d['center'][i][1:], d['center'][i][0]).tolist()
        if normalize:
            obs_length -= 1
        for k in d.keys():
            if k != 'box' and k != 'center':
                for i in range(len(d[k])):
                    d[k][i] = d[k][i][- obs_length - time_to_event:-time_to_event]
                d[k] = np.array(d[k])
            else:
                d[k] = np.array(d[k])

        d['acts'] = d['acts'][:, 0, :].copy()
        return d

    def get_model_opts(self, model_opts):
        default_opts = {'obs_input_type': ['local_box', 'local_context', 'pose', 'box', 'speed'],
                        'enlarge_ratio': 1.5,
                        'pred_target_type': ['crossing'],
                        'obs_length': 15,
                        'time_to_event': 60,
                        'dataset': 'pie',
                        'normalize_boxes': True}
        default_opts.update(model_opts)
        return default_opts

    def get_data(self, data_raw, model_opts):
        data = {}
        data_type_sizes_dict = {}

        model_opts = self.get_model_opts(model_opts)

        obs_length = model_opts['obs_length']
        time_to_event = model_opts['time_to_event']
        dataset = model_opts['dataset']
        eratio = model_opts['enlarge_ratio']
        data_type_keys = sorted(data_raw.keys())

        for k in data_type_keys:
            if k == 'test':
                data[k] = self.get_data_sequence(data_raw[k], obs_length, time_to_event, model_opts['normalize_boxes'])
            else:
                data[k] = self.get_data_sequence_balance(data_raw[k], obs_length, time_to_event, model_opts['normalize_boxes'])
            data_type_sizes_dict['box'] = data[k]['box'].shape[1:]

            if 'speed' in data[k].keys():
                data_type_sizes_dict['speed'] = data[k]['speed'].shape[1:]

            if 'pose' in model_opts['obs_input_type']:
                path_to_pose, _ = get_path(save_folder='poses',
                                           dataset=dataset,
                                           save_root_folder='data/features')
                print(path_to_pose)
                data[k]['pose'] = self.get_pose(data[k]['image'],
                                           data[k]['ped_id'], data_type=k,
                                           file_path=path_to_pose)
                data_type_sizes_dict['pose'] = data[k]['pose'].shape[1:]

            if 'local_box' in model_opts['obs_input_type']:
                print('\n#####################################')
                print('Generating local box %s' % k)
                print('#####################################')
                path_to_local_boxes, _ = get_path(save_folder='local_box',
                                                  dataset=dataset,
                                                  save_root_folder='data/features')
                data[k]['local_box'] = self.load_images_crop_and_process(data[k]['image'],
                                                                         data[k]['box_org'], data[k]['ped_id'],
                                                                         data_type=k,
                                                                         save_path=path_to_local_boxes,
                                                                         crop_type='bbox',
                                                                         crop_mode='pad_resize')
                data_type_sizes_dict['local_box'] = data[k]['local_box'].shape[1:]

            if 'local_context' in model_opts['obs_input_type']:
                print('\n#####################################')
                print('Generating local context %s' % k)
                print('#####################################')

                path_to_local_context, _ = get_path(save_folder='local_context',
                                                      dataset=dataset,
                                                      save_root_folder='data/features')
                data[k]['local_context'] = self.load_images_crop_and_process(data[k]['image'],
                                                                             data[k]['box_org'], data[k]['ped_id'],
                                                                             data_type=k,
                                                                             save_path=path_to_local_context,
                                                                             crop_type='surround',
                                                                             crop_resize_ratio=eratio)
                data_type_sizes_dict['local_context'] = data[k]['local_context'].shape[1:]

        train_test_data = {}
        data_final_keys = sorted(data.keys())
        for k in data_final_keys:
            train_test_data[k] = []

        data_sizes = []
        data_types = []

        for d_type in model_opts['obs_input_type']:
            for k in data.keys():
                train_test_data[k].append(data[k][d_type])
            data_sizes.append(data_type_sizes_dict[d_type])
            data_types.append(d_type)

        for k in data_final_keys:
            train_test_data[k] = (train_test_data[k], data[k]['acts'])

        return train_test_data, data_types, data_sizes

    def log_configs(self, config_path, batch_size, epochs, lr, opts):
        with open(config_path, 'wt') as fid:
            fid.write("####### Model options #######\n")
            for k in opts:
                fid.write("%s: %s\n" % (k, str(opts[k])))

            fid.write("\n####### Network config #######\n")
            fid.write("%s: %s\n" % ('hidden_units', str(self._num_hidden_units)))
            fid.write("%s: %s\n" % ('reg_value ', str(self._regularizer_value)))

            fid.write("\n####### Training config #######\n")
            fid.write("%s: %s\n" % ('batch_size', str(batch_size)))
            fid.write("%s: %s\n" % ('epochs', str(epochs)))
            fid.write("%s: %s\n" % ('lr', str(lr)))

        print('Wrote configs to {}'.format(config_path))

    # -- Model -------------------------------------------------------------------
    def build_model(self, data_types, data_sizes):
        return StackedGRU(data_types, data_sizes, self._num_hidden_units,
                          self._regularizer_value).to(self.device)

    # -- Train ---------------------------------------------------------------------
    def train(self, data_train, batch_size=32, epochs=60, lr=0.000005, model_opts=None):
        # PID suffix avoids collisions when multiple training processes (e.g.
        # separate experiments) start within the same second and would
        # otherwise land on the same save_folder and clobber each other's
        # checkpoint/config/history files.
        model_folder_name = time.strftime("%d%b%Y-%Hh%Mm%Ss") + '-pid%d' % os.getpid()
        model_path, model_dir = get_path(save_folder=model_folder_name,
                                         save_root_folder='data/models',
                                         file_name='model.pt')

        train_val_data, data_types, data_sizes = self.get_data({'train': data_train}, model_opts)
        train_data = train_val_data['train']

        model = self.build_model(data_types, data_sizes)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=self._regularizer_value)
        criterion = nn.BCELoss()

        inputs = [torch.from_numpy(np.asarray(x)).float() for x in train_data[0]]
        labels = torch.from_numpy(np.asarray(train_data[1])).float()

        n = labels.shape[0]
        history = {'loss': [], 'accuracy': []}
        model.train()
        for epoch in range(epochs):
            perm = torch.randperm(n)
            epoch_loss = 0.0
            epoch_correct = 0
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                batch_inputs = [x[idx].to(self.device) for x in inputs]
                batch_labels = labels[idx].to(self.device)

                optimizer.zero_grad()
                preds = model(batch_inputs).squeeze(-1)
                loss = criterion(preds, batch_labels.squeeze(-1))
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * len(idx)
                epoch_correct += ((preds > 0.5).float() == batch_labels.squeeze(-1)).sum().item()

            epoch_loss /= n
            epoch_acc = epoch_correct / n
            history['loss'].append(epoch_loss)
            history['accuracy'].append(epoch_acc)
            print('Epoch %d/%d - loss: %.4f - accuracy: %.4f' % (epoch + 1, epochs, epoch_loss, epoch_acc))

        print('Train model is saved to {}'.format(model_path))
        torch.save({
            'model_state_dict': model.state_dict(),
            'data_types': data_types,
            'data_sizes': data_sizes,
            'num_hidden_units': self._num_hidden_units,
            'regularizer_value': self._regularizer_value,
        }, model_path)

        model_opts_path, _ = get_path(save_folder=model_folder_name,
                                      save_root_folder='data/models',
                                      file_name='model_opts.pkl')
        with open(model_opts_path, 'wb') as fid:
            pickle.dump(model_opts, fid, pickle.HIGHEST_PROTOCOL)

        config_path, _ = get_path(save_folder=model_folder_name,
                                  save_root_folder='data/models',
                                  file_name='configs.txt')
        self.log_configs(config_path, batch_size, epochs, lr, model_opts)

        history_path, saved_files_path = get_path(save_folder=model_folder_name,
                                                  save_root_folder='data/models',
                                                  file_name='history.pkl')
        with open(history_path, 'wb') as fid:
            pickle.dump(history, fid, pickle.HIGHEST_PROTOCOL)

        return saved_files_path

    # -- Test ---------------------------------------------------------------------
    def test(self, data_test, model_path=''):
        with open(os.path.join(model_path, 'model_opts.pkl'), 'rb') as fid:
            model_opts = pickle.load(fid)

        checkpoint = torch.load(os.path.join(model_path, 'model.pt'), map_location=self.device)
        model = self.build_model(checkpoint['data_types'], checkpoint['data_sizes'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        test_data, _, _ = self.get_data({'test': data_test}, model_opts)
        inputs = [torch.from_numpy(np.asarray(x)).float().to(self.device) for x in test_data['test'][0]]
        labels = np.asarray(test_data['test'][1])

        with torch.no_grad():
            test_results = model(inputs).squeeze(-1).cpu().numpy()
        test_results = test_results.reshape(-1, 1)

        acc = accuracy_score(labels, np.round(test_results))
        f1 = f1_score(labels, np.round(test_results))
        # AUC must use continuous scores, not rounded 0/1 predictions --
        # rounding first collapses AUC into a near-duplicate of accuracy
        # (both driven by the same thresholded predictions), destroying its
        # value as an independent ranking-quality metric.
        auc = roc_auc_score(labels, test_results)
        roc = roc_curve(labels, test_results)
        precision = precision_score(labels, np.round(test_results))
        recall = recall_score(labels, np.round(test_results))
        pre_recall = precision_recall_curve(labels, test_results)

        print('acc:{} auc:{} f1:{} precision:{} recall:{}'.format(acc, auc, f1, precision, recall))

        save_results_path = os.path.join(model_path, '{:.2f}'.format(acc) + '.pkl')
        if not os.path.exists(save_results_path):
            results = {'results': test_results, 'data': test_data, 'acc': acc, 'auc': auc,
                      'f1': f1, 'roc': roc, 'precision': precision, 'recall': recall,
                      'pre_recall_curve': pre_recall}

        with open(save_results_path, 'wb') as fid:
            pickle.dump(test_results, fid, pickle.HIGHEST_PROTOCOL)
        return acc, auc, f1, precision, recall


class StackedGRU(nn.Module):
    """Same stacked-fusion topology as sf_gru.py:stacked_rnn — one GRU per
    modality, each modality's raw features concatenated onto the running
    hidden sequence before the next GRU, final GRU's last hidden state feeds
    a sigmoid output."""

    def __init__(self, data_types, data_sizes, hidden_units, weight_decay):
        super().__init__()
        self.data_types = data_types
        self.grus = nn.ModuleList()
        for i, size in enumerate(data_sizes):
            in_dim = size[-1] if i == 0 else hidden_units + size[-1]
            self.grus.append(nn.GRU(input_size=in_dim, hidden_size=hidden_units, batch_first=True))
        self.output = nn.Linear(hidden_units, 1)

    def forward(self, inputs):
        x = None
        for i, gru in enumerate(self.grus):
            is_last = (i == len(self.grus) - 1)
            if i == 0:
                seq_in = inputs[0]
            else:
                seq_in = torch.cat([x, inputs[i]], dim=2)
            out, h = gru(seq_in)
            x = out if not is_last else h.squeeze(0)
        return torch.sigmoid(self.output(x))
