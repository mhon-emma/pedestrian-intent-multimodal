"""
diagnostic_pie_homography.py
================================
Cheap diagnostic (not a training run) to test whether pursuing a real
geometric-calibration fix is likely to pay off, before committing to
building it. PIE publishes real camera calibration
(github.com/aras62/PIE/camera_params/calibration_data.json: K matrix,
fisheye distortion D, camera height 1270mm, pitch -10 deg) -- JAAD does
not and structurally can't have one uniform calibration (2 vehicles, 3
camera models, 5 countries). Three prior fix attempts (scale-relative
normalization, per-frame height-based pseudo-calibration, contrastive
scale-invariance) all failed to close the PIE<->JAAD cross-dataset gap
or made it worse.

This script:
  1. Builds a ground-plane homography from PIE's own published K +
     height + pitch (assuming a locally flat road plane -- standard
     for this kind of dashcam ground-plane projection).
  2. Projects each PIE box's bottom-center point (pedestrian's
     foot position -- the standard convention, since that's the point
     that's actually ON the ground plane, unlike the box center or top)
     into metric (X, Z) ground coordinates for every track in the
     observation window.
  3. Computes real-world per-frame displacement (m) and implied speed
     (m/s) and checks whether the distribution is physically plausible
     for human walking (~0.5-2 m/s) -- if projected "speeds" are
     wildly implausible, that's evidence something is wrong with the
     assumed extrinsics/plane-flatness, undermining confidence in a
     full calibration-based fix.
  4. Checks whether metric-space box statistics are MORE homogeneous
     within PIE across seed/set/split boundaries than raw-pixel-space
     statistics are (comparing set03/test vs set01+02+04/train, since
     these are different videos/routes even within the same camera
     rig) -- if metric conversion doesn't even reduce variance WITHIN
     one calibrated camera's own data, it's unlikely to help ACROSS
     two different cameras' data.

This does NOT touch JAAD at all (no calibration exists to apply) --
it's purely a PIE-internal sanity check of the calibration hypothesis.

Usage
-----
  python diagnostic_pie_homography.py

Output
------
  Printed statistics; results/pie_homography_diagnostic.pkl
"""

import logging
import os
import pickle
import sys

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

SFGRU_DIR    = '/usr1/home/mehon/emma_pedestrian-intent-multimodal'
PIE_UTIL_DIR = '/usr1/home/mehon/PIE/utilities'
PIE_DATA_DIR = '/usr1/home/mehon/data_root/pie'
RESULTS_DIR  = os.path.join(SFGRU_DIR, 'results')

sys.path.insert(0, SFGRU_DIR)
sys.path.append(PIE_UTIL_DIR)
os.chdir(SFGRU_DIR)

import train_full_pie_nospeed as _mod
from pie_data import PIE

# PIE's published calibration (github.com/aras62/PIE/camera_params/calibration_data.json)
CAM_HEIGHT_M = 1.270
CAM_PITCH_DEG = -10.0
K = np.array([
    [1004.8374471951423, 0.0, 960.1025514993675],
    [0.0, 1004.3912782107128, 573.5538287373604],
    [0.0, 0.0, 1.0],
])
D = np.array([-0.02748054291929438, -0.007055051080370751,
             -0.039625194298025156, 0.019310795479533783])
FRAME_RATE_HZ = 30.0  # PIE's native video frame rate


def build_ground_plane_homography():
    """Builds a homography mapping undistorted PIXEL coordinates
    (x, y) to GROUND-PLANE metric coordinates (X, Z) in front of the
    camera, assuming: (1) a flat local road plane, (2) the camera is
    mounted at CAM_HEIGHT_M above that plane, pitched CAM_PITCH_DEG
    downward from horizontal (negative = looking down, per PIE's
    convention), (3) zero roll and zero lateral offset (camera looks
    straight down the vehicle's forward axis).

    Standard single-camera IPM (inverse perspective mapping) derivation:
    a ground point at forward distance Z and lateral offset X projects
    to pixel (x, y) via the pinhole model combined with the known
    camera height/pitch; inverting that relationship recovers (X, Z)
    from (x, y). This is the same class of technique flagged by the
    calibration research as standard practice for road-scene IPM
    (e.g. lane-detection literature), applied here to a pedestrian's
    foot point instead of lane markings.
    """
    pitch_rad = np.deg2rad(CAM_PITCH_DEG)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    def pixel_to_ground(x, y):
        # Ray direction in camera coordinates (pinhole model, using
        # UNDISTORTED pixel coords -- see undistort_point below).
        ray_cam = np.array([(x - cx) / fx, (y - cy) / fy, 1.0])
        # Rotate by pitch: camera's optical axis is tilted DOWN by
        # |pitch| from horizontal, i.e. rotate about the camera's X
        # (horizontal) axis. PIE's convention: negative pitch = looking
        # down. Sign verified empirically: a downward-pitched camera
        # must push the horizon UP in the image, so pixels below the
        # principal point (cy) should see MORE of the ground plane, not
        # less -- using -pitch_rad made ry_rot go negative (spuriously
        # "above horizon") for ordinary in-frame pedestrian positions
        # (e.g. y=700, well below cy=573), which is physically wrong;
        # +pitch_rad reproduces the correct behavior (verified via a
        # standalone sweep of y values from image-bottom to
        # near-principal-point, checking ry_rot's sign matches "this
        # point should see ground" for all in-frame y > cy).
        cos_p, sin_p = np.cos(pitch_rad), np.sin(pitch_rad)
        # ray_cam = (rx, ry, rz); camera Y axis is "down" in image
        # convention, Z is "forward" (out of the lens).
        rx, ry, rz = ray_cam
        ry_rot = ry * cos_p - rz * sin_p
        rz_rot = ry * sin_p + rz * cos_p
        # Ground plane is at height -CAM_HEIGHT_M below the camera
        # (camera looks down at it); solve for the scale factor t such
        # that the ray's rotated Y-component reaches -CAM_HEIGHT_M.
        if ry_rot <= 1e-6:
            return None  # ray points above the horizon -- no ground intersection
        t = CAM_HEIGHT_M / ry_rot
        X = rx * t
        Z = rz_rot * t
        return X, Z

    return pixel_to_ground


def undistort_point(x, y):
    """Approximate fisheye (equidistant) undistortion using PIE's D
    coefficients, applied as a simple iterative correction around the
    principal point -- adequate for a diagnostic (not claiming
    pixel-perfect accuracy, just removing the dominant radial
    distortion so the pinhole-model ground projection isn't
    systematically biased by it)."""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x_n, y_n = (x - cx) / fx, (y - cy) / fy
    r = np.sqrt(x_n**2 + y_n**2)
    if r < 1e-9:
        return x, y
    theta = r  # small-angle approx for the equidistant model's inverse
    for _ in range(5):
        theta_d = theta * (1 + D[0] * theta**2 + D[1] * theta**4 +
                           D[2] * theta**6 + D[3] * theta**8)
        theta -= (theta_d - r) / (1 + 3 * D[0] * theta**2 + 5 * D[1] * theta**4 +
                                  7 * D[2] * theta**6 + 9 * D[3] * theta**8 + 1e-9)
    scale = theta / r
    x_u = x_n * scale * fx + cx
    y_u = y_n * scale * fy + cy
    return x_u, y_u


def project_box_to_ground(box, pixel_to_ground):
    """box = [x1, y1, x2, y2]. Uses the bottom-center point
    ((x1+x2)/2, y2) as the pedestrian's foot position -- the point
    that's actually on the ground plane, unlike the box center (which
    is at roughly hip/torso height, off the ground) or the top."""
    x_foot = (box[0] + box[2]) / 2.0
    y_foot = box[3]
    x_u, y_u = undistort_point(x_foot, y_foot)
    return pixel_to_ground(x_u, y_u)


def main():
    pixel_to_ground = build_ground_plane_homography()

    imdb = PIE(data_path=PIE_DATA_DIR)
    beh_train = imdb.generate_data_trajectory_sequence('train', **_mod.DATA_OPTS)
    beh_test = imdb.generate_data_trajectory_sequence('test', **_mod.DATA_OPTS)

    obs_length = _mod.MODEL_OPTS['obs_length']
    time_to_event = _mod.MODEL_OPTS['time_to_event']

    def compute_speeds(beh_data, label):
        all_speeds = []
        n_off_horizon = 0
        n_total = 0
        for i in range(len(beh_data['bbox'])):
            track = beh_data['bbox'][i][-obs_length - time_to_event:-time_to_event]
            if len(track) < 2:
                continue
            ground_pts = []
            for box in track:
                n_total += 1
                pt = project_box_to_ground(box, pixel_to_ground)
                if pt is None:
                    n_off_horizon += 1
                    ground_pts.append(None)
                else:
                    ground_pts.append(pt)
            for t in range(1, len(ground_pts)):
                if ground_pts[t] is None or ground_pts[t - 1] is None:
                    continue
                dx = ground_pts[t][0] - ground_pts[t - 1][0]
                dz = ground_pts[t][1] - ground_pts[t - 1][1]
                dist_m = np.sqrt(dx**2 + dz**2)
                speed_mps = dist_m * FRAME_RATE_HZ
                all_speeds.append(speed_mps)
        all_speeds = np.array(all_speeds)
        log.info('%s: %d frames total, %d (%.1f%%) projected above horizon (excluded)',
                 label, n_total, n_off_horizon, 100.0 * n_off_horizon / max(n_total, 1))
        log.info('%s: implied speed (m/s) -- mean=%.3f median=%.3f std=%.3f '
                 'p5=%.3f p95=%.3f max=%.3f (n=%d)',
                 label, all_speeds.mean(), np.median(all_speeds), all_speeds.std(),
                 np.percentile(all_speeds, 5), np.percentile(all_speeds, 95),
                 all_speeds.max(), len(all_speeds))
        frac_plausible = np.mean((all_speeds >= 0.0) & (all_speeds <= 3.0))
        log.info('%s: fraction of frame-to-frame speeds in [0, 3] m/s (plausible human '
                 'walk/light-jog range): %.1f%%', label, 100.0 * frac_plausible)
        return all_speeds

    log.info('=== Test 1: physical plausibility of projected speeds ===')
    train_speeds = compute_speeds(beh_train, 'PIE train (set01+02+04)')
    test_speeds = compute_speeds(beh_test, 'PIE test (set03)')

    log.info('=== Test 2: within-PIE homogeneity, metric vs raw-pixel ===')

    def raw_pixel_box_height_stats(beh_data, label):
        heights = []
        for i in range(len(beh_data['bbox'])):
            track = beh_data['bbox'][i][-obs_length - time_to_event:-time_to_event]
            for box in track:
                heights.append(box[3] - box[1])
        heights = np.array(heights)
        log.info('%s: raw pixel box height -- mean=%.1f std=%.1f (cv=%.3f)',
                 label, heights.mean(), heights.std(), heights.std() / heights.mean())
        return heights

    train_heights = raw_pixel_box_height_stats(beh_train, 'PIE train')
    test_heights = raw_pixel_box_height_stats(beh_test, 'PIE test')

    # Coefficient of variation (std/mean) comparison: does metric-space
    # speed vary LESS between train/test splits (different routes,
    # same camera) than raw pixel box height does? If metric conversion
    # doesn't even homogenize two splits from the SAME calibrated
    # camera, it's very unlikely to homogenize two DIFFERENT cameras
    # (PIE vs JAAD).
    train_test_speed_ratio = test_speeds.mean() / train_speeds.mean()
    train_test_height_ratio = test_heights.mean() / train_heights.mean()
    log.info('Train/test ratio -- metric speed: %.3f, raw pixel height: %.3f '
             '(closer to 1.0 = more homogeneous)',
             train_test_speed_ratio, train_test_height_ratio)

    out = os.path.join(RESULTS_DIR, 'pie_homography_diagnostic.pkl')
    with open(out, 'wb') as f:
        pickle.dump({
            'train_speeds': train_speeds, 'test_speeds': test_speeds,
            'train_heights': train_heights, 'test_heights': test_heights,
            'train_test_speed_ratio': train_test_speed_ratio,
            'train_test_height_ratio': train_test_height_ratio,
        }, f)
    log.info('Saved: %s', out)


if __name__ == '__main__':
    main()
