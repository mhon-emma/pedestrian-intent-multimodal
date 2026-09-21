"""
sf_gru_torch_contrastive.py
===============================
Fifth fix attempt for the cross-dataset generalization failure. The
first four (box exclusion, domain-adversarial, scale-relative
normalization, class-balance reweighting) all failed to close the
PIE<->JAAD gap. Domain-adversarial tried to make the WHOLE learned
representation dataset-invariant (too broad a target, and prone to the
GRL-collapse failure mode documented in train_domain_adversarial.py).
Scale-relative normalization (sf_gru_torch_scalenorm.py) divided
displacement by the pedestrian's own box size at frame 0 -- a FIXED,
one-shot rescaling -- and did not close the gap either, meaning the
box trajectory's dataset-specific structure is not just an absolute-
pixel-scale artifact removable by a single division.

This module tries a narrower, more direct target: make the model's
learned BOX-BRANCH EMBEDDING itself invariant to synthetic scale
perturbations, via a contrastive (NT-Xent-style) auxiliary loss during
training -- rather than hoping a fixed normalization formula happens
to remove exactly the dataset-specific structure. For each real
training sample, we synthesize a second "view" of the SAME underlying
box trajectory rescaled by a random factor (simulating what the same
physical motion would look like under a different camera
scale/distance), and pull that view's box-branch embedding toward the
original view's embedding (positive pair) while pushing it away from
other samples in the batch (negatives) -- explicitly training the
embedding to encode motion PATTERN rather than absolute or relative
pixel scale.

Architecture note: StackedGRU's per-modality GRUs are chained (modality
i's raw features are concatenated onto the running hidden sequence
before modality i+1's GRU), so there is no single clean "box
embedding" to extract from the stock forward pass. ContrastiveScaleGRU
instead pulls out the box GRU's OWN final hidden state right after its
own layer (before concatenation with the next modality) as the
embedding target for the contrastive loss, leaving the rest of the
forward pass (concatenation, downstream GRUs, final classification
head) structurally identical to the base StackedGRU.

Usage: see train_full_{pie,jaad}_contrastive.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sf_gru_torch import FocalBCELoss, SFGRUTorch, StackedGRU
from sklearn.metrics import roc_auc_score
import os
import pickle
import time

from utils import get_path


def _jitter_box_scale(box_window, rng, scale_range=(0.5, 1.8)):
    """box_window: a single sample's windowed box track, shape
    [obs_length, 4] = [x1,y1,x2,y2] per frame, BEFORE subtraction (same
    convention as sf_gru_torch_scalenorm.py's _scale_normalize_delta).
    Returns a synthetically rescaled COPY of the same track: the box's
    center trajectory is preserved (so the underlying "motion" is
    unchanged) but its apparent size, and thus the pixel-delta
    magnitude of any given real-world displacement, is scaled by a
    random factor -- simulating what filming the same physical
    pedestrian motion from a different distance/focal length would
    produce. This is the synthetic augmentation the contrastive loss
    is trained to be invariant to."""
    box_window = np.asarray(box_window, dtype=np.float64)
    factor = rng.uniform(*scale_range)

    cx = (box_window[:, 0] + box_window[:, 2]) / 2.0
    cy = (box_window[:, 1] + box_window[:, 3]) / 2.0
    w = (box_window[:, 2] - box_window[:, 0]) * factor
    h = (box_window[:, 3] - box_window[:, 1]) * factor

    jittered = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    return jittered


def _delta_from_window(box_window):
    """Plain frame-to-frame displacement relative to frame 0 (same as
    the base class's un-normalized branch) -- used for both the
    original and jittered views so the contrastive pair differs ONLY
    in scale, not in normalization convention."""
    box_window = np.asarray(box_window, dtype=np.float64)
    return np.subtract(box_window[1:], box_window[0]).tolist()


class ContrastiveScaleInvariantSFGRU(SFGRUTorch):
    """Overrides get_data_sequence[_balance] to also stash a
    scale-jittered second view of the box track ('box_jittered') in
    the returned dict alongside the normal 'box'. Everything else
    (windowing, center normalization, empty-sample dropping,
    class-balancing augmentation) is unchanged from the verified
    base-class / scalenorm-class source."""

    def get_data_sequence(self, data_raw, obs_length, time_to_event, normalize):
        print('\n#####################################')
        print('Generating raw data (+ scale-jittered box view)')
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

        keep = [i for i in range(len(d['acts'])) if len(d['acts'][i]) > 0]
        if len(keep) != len(d['acts']):
            print('Dropping %d empty sample(s) (indices %s)' %
                 (len(d['acts']) - len(keep), [i for i in range(len(d['acts'])) if i not in keep]))
            for k in d:
                d[k] = [d[k][i] for i in keep]

        rng = np.random.RandomState(0)
        box_jittered = []
        for i in range(len(d['box'])):
            d['box'][i] = d['box'][i][- obs_length - time_to_event:-time_to_event]
            d['center'][i] = d['center'][i][- obs_length - time_to_event:-time_to_event]
            window = np.asarray(d['box'][i])
            jittered_window = _jitter_box_scale(window, rng)
            if normalize:
                box_jittered.append(_delta_from_window(jittered_window))
                d['box'][i] = np.subtract(d['box'][i][1:], d['box'][i][0]).tolist()
                d['center'][i] = np.subtract(d['center'][i][1:], d['center'][i][0]).tolist()
            else:
                box_jittered.append(jittered_window.tolist())

        if normalize:
            obs_length -= 1

        for k in d.keys():
            if k != 'box' and k != 'center':
                for i in range(len(d[k])):
                    d[k][i] = d[k][i][- obs_length - time_to_event:-time_to_event]
                d[k] = np.array(d[k])
            else:
                d[k] = np.array(d[k])
        d['box_jittered'] = np.array(box_jittered)
        d['acts'] = d['acts'][:, 0, :]
        return d

    def get_data_sequence_balance(self, data_raw, obs_length, time_to_event, normalize):
        print('\n#####################################')
        print('Generating balanced raw data (+ scale-jittered box view)')
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

        rng = np.random.RandomState(0)
        box_jittered = []
        for i in range(len(d['box'])):
            d['box'][i] = d['box'][i][- obs_length - time_to_event:-time_to_event]
            d['center'][i] = d['center'][i][- obs_length - time_to_event:-time_to_event]
            window = np.asarray(d['box'][i])
            jittered_window = _jitter_box_scale(window, rng)
            if normalize:
                box_jittered.append(_delta_from_window(jittered_window))
                d['box'][i] = np.subtract(d['box'][i][1:], d['box'][i][0]).tolist()
                d['center'][i] = np.subtract(d['center'][i][1:], d['center'][i][0]).tolist()
            else:
                box_jittered.append(jittered_window.tolist())
        if normalize:
            obs_length -= 1
        for k in d.keys():
            if k != 'box' and k != 'center':
                for i in range(len(d[k])):
                    d[k][i] = d[k][i][- obs_length - time_to_event:-time_to_event]
                d[k] = np.array(d[k])
            else:
                d[k] = np.array(d[k])

        d['box_jittered'] = np.array(box_jittered)
        d['acts'] = d['acts'][:, 0, :].copy()
        return d

    def build_model(self, data_types, data_sizes):
        return ContrastiveScaleGRU(data_types, data_sizes, self._num_hidden_units,
                                   self._regularizer_value).to(self.device)

    def train(self, data_train, data_val=None, batch_size=32, epochs=60, lr=0.000005,
             model_opts=None, focal_alpha=0.5, contrastive_weight=0.5, contrastive_temp=0.2):
        """Same structure as SFGRUTorch.train, plus: (1) box_jittered is
        pulled out of get_data's per-split dict as its own model input
        (NOT passed to the classifier -- only used for the contrastive
        term), and (2) each batch's loss is
        classification_loss + contrastive_weight * NT-Xent(box_embed,
        box_embed_jittered)."""
        model_folder_name = time.strftime("%d%b%Y-%Hh%Mm%Ss") + '-pid%d' % os.getpid()
        model_path, model_dir = get_path(save_folder=model_folder_name,
                                         save_root_folder='data/models',
                                         file_name='model.pt')

        data_dict = {'train': data_train}
        if data_val is not None:
            data_dict['val'] = data_val
        train_val_data, data_types, data_sizes = self.get_data(data_dict, model_opts)

        box_idx = data_types.index('box')

        train_data = train_val_data['train']
        val_data = train_val_data['val'] if data_val is not None else None

        # box_jittered was computed during get_data_sequence[_balance] but
        # get_data (base class, unmodified) doesn't know about it -- pull
        # it back out of the raw per-split dicts we already computed.
        train_box_jittered = self._last_box_jittered['train']
        val_box_jittered = self._last_box_jittered.get('val') if data_val is not None else None

        model = self.build_model(data_types, data_sizes)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=self._regularizer_value)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
        criterion = FocalBCELoss(alpha=focal_alpha, gamma=1.0)

        inputs = [torch.from_numpy(np.asarray(x)).float() for x in train_data[0]]
        labels = torch.from_numpy(np.asarray(train_data[1])).float()
        box_jittered_t = torch.from_numpy(np.asarray(train_box_jittered)).float()

        if val_data is not None:
            val_inputs = [torch.from_numpy(np.asarray(x)).float().to(self.device) for x in val_data[0]]
            val_labels = torch.from_numpy(np.asarray(val_data[1])).float().to(self.device)

        best_val_auc = -float('inf')
        best_epoch = -1

        n = labels.shape[0]
        history = {'loss': [], 'cls_loss': [], 'contrastive_loss': [], 'accuracy': [],
                  'val_loss': [], 'val_accuracy': [], 'val_auc': []}
        for epoch in range(epochs):
            model.train()
            perm = torch.randperm(n)
            epoch_loss = 0.0
            epoch_cls_loss = 0.0
            epoch_contrastive_loss = 0.0
            epoch_correct = 0
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                batch_inputs = [x[idx].to(self.device) for x in inputs]
                batch_labels = labels[idx].to(self.device)
                batch_box_jittered = box_jittered_t[idx].to(self.device)

                optimizer.zero_grad()
                preds, box_embed = model(batch_inputs, return_box_embedding=True)
                preds = preds.squeeze(-1)
                cls_loss = criterion(preds, batch_labels.squeeze(-1))

                jittered_inputs = list(batch_inputs)
                jittered_inputs[box_idx] = batch_box_jittered
                _, box_embed_jittered = model(jittered_inputs, return_box_embedding=True)

                contrastive_loss = _nt_xent_loss(box_embed, box_embed_jittered, temperature=contrastive_temp)

                loss = cls_loss + contrastive_weight * contrastive_loss
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * len(idx)
                epoch_cls_loss += cls_loss.item() * len(idx)
                epoch_contrastive_loss += contrastive_loss.item() * len(idx)
                epoch_correct += ((preds > 0.5).float() == batch_labels.squeeze(-1)).sum().item()

            epoch_loss /= n
            epoch_cls_loss /= n
            epoch_contrastive_loss /= n
            epoch_acc = epoch_correct / n
            history['loss'].append(epoch_loss)
            history['cls_loss'].append(epoch_cls_loss)
            history['contrastive_loss'].append(epoch_contrastive_loss)
            history['accuracy'].append(epoch_acc)

            if val_data is not None:
                model.eval()
                with torch.no_grad():
                    val_preds = model(val_inputs).squeeze(-1)
                    val_loss = criterion(val_preds, val_labels.squeeze(-1)).item()
                    val_acc = ((val_preds > 0.5).float() == val_labels.squeeze(-1)).sum().item() / val_labels.size(0)
                    val_auc = roc_auc_score(val_labels.cpu().numpy(), val_preds.cpu().numpy())
                    history['val_loss'].append(val_loss)
                    history['val_accuracy'].append(val_acc)
                    history['val_auc'].append(val_auc)

                    if val_auc > best_val_auc:
                        best_val_auc = val_auc
                        best_epoch = epoch
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'data_types': data_types,
                            'data_sizes': data_sizes,
                            'num_hidden_units': self._num_hidden_units,
                            'regularizer_value': self._regularizer_value,
                        }, model_path)
                        print('Best model saved at epoch %d with val_auc: %.4f' % (epoch + 1, best_val_auc))
                scheduler.step(val_loss)
                print('Epoch %d/%d - loss: %.4f (cls: %.4f, contrastive: %.4f) - accuracy: %.4f - '
                     'val_loss: %.4f - val_accuracy: %.4f - val_auc: %.4f' %
                     (epoch + 1, epochs, epoch_loss, epoch_cls_loss, epoch_contrastive_loss,
                      epoch_acc, val_loss, val_acc, val_auc))
            else:
                scheduler.step(epoch_loss)
                print('Epoch %d/%d - loss: %.4f (cls: %.4f, contrastive: %.4f) - accuracy: %.4f' %
                     (epoch + 1, epochs, epoch_loss, epoch_cls_loss, epoch_contrastive_loss, epoch_acc))

        if val_data is not None:
            print('Best model was at epoch %d with val_auc: %.4f' % (best_epoch + 1, best_val_auc))
        else:
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

        history_path, _ = get_path(save_folder=model_folder_name,
                                   save_root_folder='data/models',
                                   file_name='history.pkl')
        with open(history_path, 'wb') as fid:
            pickle.dump(history, fid, pickle.HIGHEST_PROTOCOL)

        return model_dir

    def get_data(self, data_raw, model_opts):
        """Wraps the base class's get_data: after each split's
        get_data_sequence[_balance] call populates a dict with
        'box_jittered', that key is consumed here (stashed on self,
        since the base class's get_data doesn't know about it and
        would otherwise try to treat it as another obs_input_type
        modality) before delegating to the unmodified base
        implementation for everything else."""
        obs_length = model_opts.get('obs_length', 15)
        time_to_event = model_opts.get('time_to_event', 60)
        normalize = model_opts.get('normalize_boxes', True)

        self._last_box_jittered = {}
        for k, raw in data_raw.items():
            if k == 'test':
                d = self.get_data_sequence(raw, obs_length, time_to_event, normalize)
            else:
                d = self.get_data_sequence_balance(raw, obs_length, time_to_event, normalize)
            self._last_box_jittered[k] = d['box_jittered']

        return super().get_data(data_raw, model_opts)


def _nt_xent_loss(z1, z2, temperature=0.2):
    """Standard NT-Xent (normalized temperature-scaled cross-entropy)
    contrastive loss, SimCLR-style: for each sample i, z1[i] and z2[i]
    are the two "views" (original-scale and jittered-scale box
    embeddings) of the SAME underlying trajectory -- the positive pair.
    All other 2N-2 embeddings in the batch (both views' embeddings for
    every other sample) are negatives. Minimizing this loss pulls
    positive pairs together and pushes negatives apart in embedding
    space, which is exactly "make the box embedding invariant to scale
    while still distinguishing genuinely different trajectories" --
    the scale-invariance target this experiment is testing, in
    contrast to scale-relative normalization's fixed, non-learned
    rescaling."""
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    batch_size = z1.shape[0]

    z = torch.cat([z1, z2], dim=0)  # [2N, D]
    sim = torch.matmul(z, z.T) / temperature  # [2N, 2N]

    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, -1e9)

    pos_idx = torch.arange(2 * batch_size, device=z.device)
    pos_idx = (pos_idx + batch_size) % (2 * batch_size)

    return F.cross_entropy(sim, pos_idx)


class ContrastiveScaleGRU(StackedGRU):
    """Same forward pass as StackedGRU, but optionally also returns the
    box modality's own GRU final hidden state (before it gets
    concatenated onto the next modality's input) as a standalone
    embedding for the contrastive loss."""

    def forward(self, inputs, return_box_embedding=False):
        x = None
        box_embed = None
        for i, gru in enumerate(self.grus):
            is_last = (i == len(self.grus) - 1)
            if i == 0:
                seq_in = inputs[0]
            else:
                seq_in = torch.cat([x, inputs[i]], dim=2)
            out, h = gru(seq_in)
            if self.data_types[i] == 'box':
                box_embed = h.squeeze(0)
            x = out if not is_last else h.squeeze(0)
        preds = torch.sigmoid(self.output(x))
        if return_box_embedding:
            return preds, box_embed
        return preds
