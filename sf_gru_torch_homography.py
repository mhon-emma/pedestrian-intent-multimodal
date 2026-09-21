"""
sf_gru_torch_homography.py
==============================
Seventh fix attempt for the cross-dataset generalization failure.
diagnostic_pie_homography.py verified (after fixing a real sign-error
bug in the pitch-rotation math) that projecting PIE's box trajectories
through a ground-plane homography built from PIE's OWN PUBLISHED
CALIBRATION (github.com/aras62/PIE/camera_params/calibration_data.json:
K matrix, fisheye distortion D, camera height 1.270m, pitch -10 deg)
produces physically plausible pedestrian speeds (100% of frame-to-frame
speeds in the [0,3] m/s human walk/light-jog range, vs 40% before the
bug fix) and homogenizes PIE's own train/test splits BETTER than raw
pixel space does (metric speed ratio 0.961 vs raw-pixel-height ratio
0.909, closer to 1.0 = more homogeneous). This is the first fix
attempt this session with a genuinely positive diagnostic signal before
committing to a full training run -- the three prior attempts
(scale-relative normalization, per-frame height-based pseudo-
calibration applied to BOTH datasets uniformly, contrastive
scale-invariance) either failed to move the cross-dataset gap or made
it worse.

JAAD has no official calibration and structurally can't have one
uniform value (2 vehicles, 3 consumer dashcam models -- GoPro HERO+,
Garmin GDR-35, Highscreen Black Box -- across 5 countries). Rather than
skip JAAD, this module applies the SAME ground-plane-homography
MACHINERY to JAAD using an ESTIMATED default height/pitch appropriate
for a windshield-mounted consumer dashcam (see JAAD_CAM_HEIGHT_M /
JAAD_CAM_PITCH_DEG below and their sourcing comment) -- an approximate
single-camera stand-in, not real per-video calibration, but a
principled physics-based projection rather than a purely empirical
rescaling (which is what every previous fix attempt used).

Implementation follows the sf_gru_torch_scalenorm.py /
sf_gru_torch_heightnorm.py pattern: full override of
get_data_sequence[_balance], since box_org's frame-alignment issue
(documented in sf_gru_torch_scalenorm.py) means computing the ground
projection must happen against the box's own pre-subtraction window,
not the separately-windowed box_org.
"""

import numpy as np

from sf_gru_torch import SFGRUTorch

# ---------------------------------------------------------------------------
# PIE: real published calibration
# (github.com/aras62/PIE/camera_params/calibration_data.json)
# ---------------------------------------------------------------------------
PIE_CAM_HEIGHT_M = 1.270
PIE_CAM_PITCH_DEG = -10.0
PIE_K = np.array([
    [1004.8374471951423, 0.0, 960.1025514993675],
    [0.0, 1004.3912782107128, 573.5538287373604],
    [0.0, 0.0, 1.0],
])
PIE_D = np.array([-0.02748054291929438, -0.007055051080370751,
                  -0.039625194298025156, 0.019310795479533783])

# ---------------------------------------------------------------------------
# JAAD: no official calibration exists (2 vehicles, 3 consumer dashcam
# models across 5 countries -- structurally cannot have one uniform
# value). Estimated defaults for a windshield/mirror-mounted consumer
# dashcam (GoPro HERO+ / Garmin GDR-35 / Highscreen Black Box), sourced
# from a dedicated research pass on typical consumer dashcam mounting
# practice and windshield/mirror geometry: height ~1.40m (windshield
# mount, near mirror height on a typical sedan/crossover -- somewhat
# HIGHER than PIE's 1.27m, since JAAD's cameras sit near the mirror
# rather than PIE's lower, more centrally-mounted research rig), pitch
# ~-5 deg (consumer installers tilt only slightly downward to keep
# ~2/3 road, 1/3 sky in frame at near-eye-level mount height, unlike
# PIE's more aggressively-pitched -10 deg purpose-built AV research
# rig). No prior published work assumes explicit JAAD extrinsics for
# ground-plane/IPM -- these are a documented estimate with real
# uncertainty (height 1.2-1.5m, pitch -3 to -8 deg plausible range),
# not a literature-sourced constant. JAAD's own image resolution/FOV
# also differs by camera model (1920x1080 vs 1280x720); we approximate
# with PIE's own K matrix rescaled to JAAD's frame width, since no
# per-camera intrinsics exist either -- this is explicitly a rough,
# single approximate camera, not a claim of per-clip accuracy.
# ---------------------------------------------------------------------------
JAAD_CAM_HEIGHT_M = 1.40
JAAD_CAM_PITCH_DEG = -5.0
JAAD_FRAME_WIDTH = 1920  # most JAAD clips; rescale K if a clip differs
_pie_to_jaad_scale = JAAD_FRAME_WIDTH / 1920.0
JAAD_K = PIE_K.copy()
JAAD_K[0, 0] *= _pie_to_jaad_scale
JAAD_K[1, 1] *= _pie_to_jaad_scale
JAAD_K[0, 2] *= _pie_to_jaad_scale
JAAD_K[1, 2] *= _pie_to_jaad_scale
JAAD_D = np.array([0.0, 0.0, 0.0, 0.0])  # no distortion correction (unknown lens)

FRAME_RATE_HZ = 30.0


def _make_ground_projector(cam_height_m, cam_pitch_deg, K):
    """Returns a pixel_to_ground(x, y) closure -- see
    diagnostic_pie_homography.py's build_ground_plane_homography for
    the full derivation and the sign-convention verification (pitch
    rotation uses +pitch_rad, confirmed empirically against expected
    in-frame horizon behavior)."""
    pitch_rad = np.deg2rad(cam_pitch_deg)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    cos_p, sin_p = np.cos(pitch_rad), np.sin(pitch_rad)

    def pixel_to_ground(x, y):
        rx = (x - cx) / fx
        ry = (y - cy) / fy
        rz = 1.0
        ry_rot = ry * cos_p - rz * sin_p
        rz_rot = ry * sin_p + rz * cos_p
        if ry_rot <= 1e-6:
            return None
        t = cam_height_m / ry_rot
        return rx * t, rz_rot * t

    return pixel_to_ground


def _make_undistorter(K, D):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    def undistort_point(x, y):
        if np.all(D == 0):
            return x, y
        x_n, y_n = (x - cx) / fx, (y - cy) / fy
        r = np.sqrt(x_n**2 + y_n**2)
        if r < 1e-9:
            return x, y
        theta = r
        for _ in range(5):
            theta_d = theta * (1 + D[0] * theta**2 + D[1] * theta**4 +
                               D[2] * theta**6 + D[3] * theta**8)
            theta -= (theta_d - r) / (1 + 3 * D[0] * theta**2 + 5 * D[1] * theta**4 +
                                      7 * D[2] * theta**6 + 9 * D[3] * theta**8 + 1e-9)
        scale = theta / r
        return x_n * scale * fx + cx, y_n * scale * fy + cy

    return undistort_point


_pie_ground = _make_ground_projector(PIE_CAM_HEIGHT_M, PIE_CAM_PITCH_DEG, PIE_K)
_pie_undistort = _make_undistorter(PIE_K, PIE_D)
_jaad_ground = _make_ground_projector(JAAD_CAM_HEIGHT_M, JAAD_CAM_PITCH_DEG, JAAD_K)
_jaad_undistort = _make_undistorter(JAAD_K, JAAD_D)

# Fallback for off-horizon / degenerate projections: rather than drop
# the frame (which would create ragged sequences the GRU can't
# consume), fall back to the RAW pixel displacement for that one
# frame -- a rare edge case (distant pedestrians near the horizon;
# empirically <1% of frames after the sign fix), not the dominant
# signal.
_FALLBACK_PIXEL_SCALE = 0.01  # rough px->m scale for the fallback case only


def _project_box_foot(box, undistort_fn, ground_fn):
    x_foot = (box[0] + box[2]) / 2.0
    y_foot = box[3]
    xu, yu = undistort_fn(x_foot, y_foot)
    pt = ground_fn(xu, yu)
    if pt is None:
        return (x_foot * _FALLBACK_PIXEL_SCALE, y_foot * _FALLBACK_PIXEL_SCALE)
    return pt


def _ground_project_delta(box_window, undistort_fn, ground_fn):
    """box_window: [obs_length, 4] raw pixel boxes, BEFORE subtraction.
    Returns metric-space (X, Z) deltas relative to frame 0, shape
    [obs_length-1, 2] -- NOTE: 2 dims (X, Z), not 4 like the raw
    box's [dx1,dy1,dx2,dy2]. The box's SIZE information (width/height,
    which raw pixel delta implicitly carries via x2-x1/y2-y1 changing)
    is not geometrically meaningful in ground-plane space the same
    way -- ground projection is specifically about the FOOT POINT's
    real-world trajectory, not a re-parameterized box shape. We
    concatenate a constant-zero pair to keep the feature dimensionality
    compatible with the rest of the pipeline (4 dims), so this fix can
    be swapped in without touching build_model/data_sizes plumbing."""
    ground_pts = [_project_box_foot(box, undistort_fn, ground_fn) for box in box_window]
    ground_pts = np.array(ground_pts)  # [obs_length, 2]
    deltas = np.subtract(ground_pts[1:], ground_pts[0])  # [obs_length-1, 2]
    padding = np.zeros((deltas.shape[0], 2))
    return np.concatenate([deltas, padding], axis=1).tolist()


class HomographySFGRUTorch(SFGRUTorch):
    """PIE-calibrated / JAAD-estimated ground-plane-projected box
    feature. Which projector to use is selected via
    HomographySFGRUTorch.DATASET (set by the training script before
    calling train()/test()) since get_data_sequence[_balance] don't
    otherwise know which dataset they're processing."""

    DATASET = 'pie'  # overridden by train_full_{pie,jaad}_homography.py

    # Optional instance-level override for the JAAD projector, used by
    # the camera-parameter sensitivity sweep (sweep_jaad_camera_params.py)
    # to test different height/pitch assumptions at EVAL time without
    # retraining or touching the module-level defaults (JAAD_CAM_HEIGHT_M
    # / JAAD_CAM_PITCH_DEG) that every other script relies on.
    _jaad_projector_override = None  # (undistort_fn, ground_fn) tuple, or None

    def _undistort_and_ground(self):
        if self.DATASET == 'jaad':
            if self._jaad_projector_override is not None:
                return self._jaad_projector_override
            return _jaad_undistort, _jaad_ground
        return _pie_undistort, _pie_ground

    def get_data_sequence(self, data_raw, obs_length, time_to_event, normalize):
        print('\n#####################################')
        print('Generating raw data (ground-plane-projected box, dataset=%s)' % self.DATASET)
        print('#####################################')
        undistort_fn, ground_fn = self._undistort_and_ground()

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
                d['box'][i] = _ground_project_delta(d['box'][i], undistort_fn, ground_fn)
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
        print('Generating balanced raw data (ground-plane-projected box, dataset=%s)' % self.DATASET)
        print('#####################################')
        undistort_fn, ground_fn = self._undistort_and_ground()

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
                d['box'][i] = _ground_project_delta(d['box'][i], undistort_fn, ground_fn)
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
