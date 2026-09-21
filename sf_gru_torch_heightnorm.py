"""
sf_gru_torch_heightnorm.py
==============================
Sixth fix attempt for the cross-dataset generalization failure. Camera-
calibration research found PIE has official calibration (full K matrix,
fisheye distortion, camera height 1270mm, pitch -10 deg -- see
github.com/aras62/PIE/camera_params/calibration_data.json) but JAAD has
NONE and structurally can't have one uniform calibration (2 vehicles, 3
camera models, 5 countries/locations, 2 resolutions). A full per-video
homography fix is therefore only cleanly possible for PIE, not JAAD,
which blocks a real cross-dataset test.

This module implements a cheap, dataset-agnostic proxy instead: use the
pedestrian's own apparent box HEIGHT in pixels, combined with a
standard adult height prior (~1.7m), as a single-point pseudo-
calibration per frame -- a lightweight stand-in for true single-view
metrology. Under a simple pinhole approximation, a pedestrian's real
height H (constant, ~1.7m) projects to pixel height h(t) =
f * H / depth(t), so depth(t) ~ 1/h(t) (up to the unknown but per-frame-
CONSTANT focal length f, which cancels out of relative comparisons
within a track). Box position/displacement is then rescaled using this
implied depth-proportional factor, rather than the box's own raw pixel
extent (which is what sf_gru_torch_scalenorm.py's fixed first-frame
normalization did, and which did not close the gap) -- the difference
being that this normalization varies FRAME BY FRAME as the pedestrian's
apparent distance changes, not just once per track.

Why try this before full homography: it needs no per-video landmark
annotation, no vanishing-point estimation, and no assumption about
JAAD's (unknown, heterogeneous) camera intrinsics -- only the box
height itself, which is already an input to every architecture tested
this session. If this ALSO fails to close the gap, that's evidence the
problem isn't fixable by any single-frame/single-track scale correction
(pixel or metric) and something else (labeling protocol, behavioral
distribution, road geometry) is likely the dominant remaining factor --
directly informing whether the heavier PIE-only real-homography
diagnostic is worth the additional engineering time.
"""

import numpy as np

from sf_gru_torch import SFGRUTorch

_ASSUMED_ADULT_HEIGHT_M = 1.7
_EPS_PX = 1.0  # avoid divide-by-zero for degenerate (near-zero-height) boxes


def _height_normalize_delta(box_window):
    """box_window: a single sample's windowed box track, shape
    [obs_length, 4] = [x1,y1,x2,y2] per frame, BEFORE subtraction (same
    convention as sf_gru_torch_scalenorm.py's _scale_normalize_delta).

    Unlike scale-relative normalization (which divides every frame's
    displacement by frame 0's box size -- a single, track-constant
    divisor), this computes a PER-FRAME implied depth-proportional
    factor from that frame's own box height, then expresses each
    frame's raw pixel position in those depth-scaled units before
    taking the delta. This captures within-track perspective changes
    (the pedestrian getting closer/farther during the observation
    window) that a single first-frame divisor cannot."""
    box_window = np.asarray(box_window, dtype=np.float64)
    heights_px = np.maximum(box_window[:, 3] - box_window[:, 1], _EPS_PX)

    # depth_proxy(t) ~ 1 / height_px(t) (up to a constant focal length
    # that cancels in relative units -- we only need INTERNAL
    # consistency within a track, not an absolute metric value, since
    # the goal is comparability across TRACKS/DATASETS, achieved by
    # anchoring every track to the same real-world constant (adult
    # height) rather than each dataset's own arbitrary pixel scale).
    depth_proxy = 1.0 / heights_px

    # Re-express each frame's box corners in "depth-normalized" pixel
    # units: position * depth_proxy(t) -- this is the standard
    # perspective-projection move (position scales inversely with
    # depth for a fixed real-world size), applied per-frame rather
    # than once per track.
    normalized = box_window * depth_proxy[:, None]

    raw_delta = np.subtract(normalized[1:], normalized[0])
    return raw_delta.tolist()


class HeightNormSFGRUTorch(SFGRUTorch):
    """Identical to SFGRUTorch.get_data_sequence[_balance] except the
    normalize branch's box subtraction is replaced with
    _height_normalize_delta. Every other line -- windowing, center
    normalization, empty-sample dropping, class-balancing augmentation --
    is unchanged from the verified base-class source (same pattern as
    sf_gru_torch_scalenorm.py)."""

    def get_data_sequence(self, data_raw, obs_length, time_to_event, normalize):
        print('\n#####################################')
        print('Generating raw data (height-normalized box)')
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
                d['box'][i] = _height_normalize_delta(d['box'][i])
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
        print('Generating balanced raw data (height-normalized box)')
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
                d['box'][i] = _height_normalize_delta(d['box'][i])
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
