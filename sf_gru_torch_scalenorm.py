"""
sf_gru_torch_scalenorm.py
============================
Alternative fix for the cross-dataset generalization failure diagnosed in
cross_dataset_modality_ablation.py: excluding box entirely
(train_full_{pie,jaad}_noboxspeed.py) costs real in-domain accuracy on
PIE, and domain-adversarial training (train_domain_adversarial.py), once
its GRL-collapse failure mode was fixed (domain_loss_weight 0.1 instead
of 1.0), did not outperform the simpler box-exclusion baseline while
being far more complex to tune correctly.

This module tests a cheaper, more directly targeted hypothesis: the raw
bounding-box displacement feature is

    box[i][t] = box_org[i][t] - box_org[i][0]   (per-frame [dx1,dy1,dx2,dy2]
                                                   in RAW PIXEL units,
                                                   relative to the window's
                                                   first frame)

so the same physical pedestrian motion produces different pixel-delta
magnitudes across PIE and JAAD if the two datasets differ in camera
resolution, mounting height, or focal length -- a direct, mechanical
explanation for box's dataset-specificity. We normalize each frame's
displacement by the pedestrian's OWN apparent box size at the window's
first frame (a standard move in trajectory literature to remove
camera-intrinsic scale dependence): scale-normalized displacement =
raw pixel displacement / [box_width_0, box_height_0, box_width_0, box_height_0].

Implementation note -- why this is a full override, not a post-hoc patch
--------------------------------------------------------------------------
sf_gru_torch.py's get_data_sequence[_balance] stores an auxiliary
'box_org' key, but it is NOT a frame-aligned copy of the pre-subtraction
box window: 'box_org' is assigned from the FULL, unwindowed track before
get_data_sequence's own windowing loop runs, then windowed SEPARATELY,
afterward, using an already-decremented obs_length (decremented because
normalize=True subtracts one frame). This makes 'box_org' end up shifted
by one frame relative to 'box's own pre-subtraction 15-frame window --
confirmed by direct inspection (manually reconstructing box_org[1:] -
box_org[0] does not exactly reproduce the stored box[] values). Relying
on 'box_org' for per-frame scale would silently misalign scale factors
with displacements by one frame. To avoid this, ScaleNormSFGRUTorch
fully overrides get_data_sequence and get_data_sequence_balance,
replicating the base class's logic exactly (verified line-for-line
against sf_gru_torch.py) except that the normalize branch divides by
the window's own first-frame box dimensions at the moment the delta is
computed -- guaranteed alignment since it comes from the exact same
array slice used for the subtraction itself.
"""

import numpy as np

from sf_gru_torch import SFGRUTorch

_EPS = 1e-3  # avoid divide-by-zero for degenerate (near-zero-area) boxes


def _scale_normalize_delta(box_window):
    """box_window: a single sample's windowed box track, shape
    [obs_length, 4] = [x1,y1,x2,y2] per frame, BEFORE subtraction.
    Returns scale-normalized deltas, shape [obs_length-1, 4], where frame
    t's [dx1,dy1,dx2,dy2] (relative to frame 0) is divided by frame 0's
    own [width,height,width,height] -- the same reference frame the raw
    displacement is computed relative to, so scale and displacement are
    guaranteed frame-aligned by construction."""
    box_window = np.asarray(box_window, dtype=np.float64)
    raw_delta = np.subtract(box_window[1:], box_window[0])  # [obs_length-1, 4]

    x1_0, y1_0, x2_0, y2_0 = box_window[0]
    width0 = max(x2_0 - x1_0, _EPS)
    height0 = max(y2_0 - y1_0, _EPS)
    scale = np.array([width0, height0, width0, height0])

    return (raw_delta / scale).tolist()


class ScaleNormSFGRUTorch(SFGRUTorch):
    """Identical to SFGRUTorch.get_data_sequence[_balance] except the
    normalize branch's box subtraction is replaced with
    _scale_normalize_delta. Every other line -- windowing, center
    normalization, empty-sample dropping, class-balancing augmentation --
    is unchanged from the verified base-class source."""

    def get_data_sequence(self, data_raw, obs_length, time_to_event, normalize):
        print('\n#####################################')
        print('Generating raw data (scale-normalized box)')
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

        for i in range(len(d['box'])):
            d['box'][i] = d['box'][i][- obs_length - time_to_event:-time_to_event]
            d['center'][i] = d['center'][i][- obs_length - time_to_event:-time_to_event]
            if normalize:
                d['box'][i] = _scale_normalize_delta(d['box'][i])
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
        print('Generating balanced raw data (scale-normalized box)')
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

        for i in range(len(d['box'])):
            d['box'][i] = d['box'][i][- obs_length - time_to_event:-time_to_event]
            d['center'][i] = d['center'][i][- obs_length - time_to_event:-time_to_event]
            if normalize:
                d['box'][i] = _scale_normalize_delta(d['box'][i])
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
