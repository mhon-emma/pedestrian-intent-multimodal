"""
synthetic_speed.py
=====================
Investigation #1 (missing modality): PIE has continuous OBD ego-vehicle
speed; JAAD has none -- every script in this project so far has
excluded 'speed' entirely from BOTH datasets just to keep the modality
sets comparable (train_full_{pie,jaad}_nospeed.py and everything built
on them), meaning PIE's model never gets to use a real signal it
natively has, purely to accommodate JAAD's gap. This module estimates
JAAD's missing ego-vehicle speed from raw video via dense optical flow,
so JAAD can be given a genuine 'obd_speed' key and both datasets can
train with the ORIGINAL, non-reduced modality set (local_box,
local_context, pose, box, speed) instead of the deliberately narrowed
one used everywhere else in this project.

Method: for each consecutive frame pair in a pedestrian's track window,
compute dense optical flow (Farneback -- fast, no model download,
matches this being a lightweight estimate rather than a precise one)
over the REGION EXCLUDING the pedestrian's own bounding box (background
motion approximates ego-vehicle motion; the pedestrian's own motion in
frame is a confound we deliberately exclude, mirroring
sf_gru_torch_finetuned_context.py's 'surround' crop masking the
pedestrian out for the same reason). Reduce to a single scalar per
frame pair: the median flow magnitude over the excluded-bbox region
(median, not mean, for robustness against other moving pedestrians/
vehicles in frame, which would inflate a mean but not a median as long
as they don't dominate the frame).

This produces a RELATIVE, unitless ego-motion proxy, not calibrated
km/h -- there is no ground truth to calibrate against for JAAD (that is
exactly the missing information this module exists to approximate).
The synthesized 'obd_speed' values are on a different scale than PIE's
real km/h readings; z-score normalizing each dataset's speed channel
independently before use (matching this project's normalize_boxes=True
convention for other modalities) accounts for this scale mismatch
without needing to guess an absolute conversion factor.

Caching: computed once per video (not per pedestrian -- ego-motion is a
property of the video/frame pair, shared across all pedestrians visible
in it), keyed by frame path pairs, and cached to disk since Farneback
flow over full-resolution frames is the slow part of this pipeline.
"""

import hashlib
import os
import pickle

import cv2
import numpy as np

SPEED_CACHE_DIR = '/usr1/home/mehon/emma_pedestrian-intent-multimodal/data/features/jaad_synthetic_speed'

# JAAD frames are full 1080p -- Farneback dense flow at that resolution
# is far more expensive than this coarse relative ego-motion estimate
# needs (confirmed empirically: a dry run computing speed for only the
# 103-sample JAAD train split took >35 min of CPU time and was still
# not finished at that point). Downsampling to a fixed small resolution
# before computing flow cuts Farneback's cost roughly quadratically in
# the linear downsample factor, while still capturing the same coarse
# background-motion signal this estimate is after -- we only need a
# relative speed proxy, not pixel-accurate flow.
FLOW_RESIZE_WIDTH = 320


def _frame_pair_cache_key(prev_path, curr_path):
    h = hashlib.sha1(f'{prev_path}|{curr_path}'.encode()).hexdigest()
    return h


def _compute_flow_magnitude(prev_gray, curr_gray, exclude_bbox=None):
    """Farneback dense optical flow between two grayscale frames
    (already downsampled to FLOW_RESIZE_WIDTH by the caller), returns
    the median flow magnitude over the region OUTSIDE exclude_bbox (the
    pedestrian's own box, [x1,y1,x2,y2] ints, already scaled to match
    the downsampled frame by the caller) -- if exclude_bbox is None or
    covers the whole frame, falls back to the full-frame median (rare
    edge case, a track right at frame boundaries)."""
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=2, winsize=15, iterations=2,
        poly_n=5, poly_sigma=1.1, flags=0)
    magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)

    if exclude_bbox is not None:
        h, w = magnitude.shape
        x1, y1, x2, y2 = [int(round(v)) for v in exclude_bbox]
        x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
        y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
        mask = np.ones((h, w), dtype=bool)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = False
        if mask.sum() > 0:
            return float(np.median(magnitude[mask]))

    return float(np.median(magnitude))


def _load_downsampled_gray(path):
    """Loads a frame and downsamples to FLOW_RESIZE_WIDTH, preserving
    aspect ratio. Returns (gray_image, scale_factor) -- scale_factor is
    downsampled_width / original_width, needed to rescale bboxes into
    the same coordinate space."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None
    h, w = img.shape
    scale = FLOW_RESIZE_WIDTH / w
    new_size = (FLOW_RESIZE_WIDTH, max(1, int(round(h * scale))))
    small = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    return small, scale


def estimate_speed_for_track(img_paths, bboxes, cache=True):
    """img_paths: list of frame paths for one pedestrian track window
    (same ordering/length as the 'image' sequence get_data_sequence
    consumes). bboxes: list of [x1,y1,x2,y2] boxes (ORIGINAL, full-
    resolution coordinates -- this function downsamples internally and
    rescales the bbox to match), same length, the pedestrian's OWN box
    per frame (excluded from the flow computation). Returns a list of
    length len(img_paths), same convention as PIE's obd_speed
    ([[v0], [v1], ...] per-frame scalar list) -- first frame gets the
    same value as the second (no prior frame to diff against, matching
    how a real speedometer reading would just persist rather than being
    undefined).

    Every frame is loaded+downsampled exactly ONCE regardless of cache
    hits (a cache hit for the flow MAGNITUDE still needs the downsampled
    image available as next iteration's prev_gray -- the earlier version
    of this function re-read the FULL-resolution image on every cache
    hit just for that, doubling the exact cost this downsampling change
    is meant to eliminate; fixed here by always loading+downsampling
    once per frame regardless of cache state)."""
    if cache:
        os.makedirs(SPEED_CACHE_DIR, exist_ok=True)

    speeds = []
    prev_gray = None
    for i, (path, bbox) in enumerate(zip(img_paths, bboxes)):
        gray, scale = _load_downsampled_gray(path)
        if gray is None:
            # Missing/corrupt frame -- fall back to the previous speed
            # value (or 0.0 if this is the first frame) rather than
            # crashing the whole track's speed estimation. prev_gray is
            # deliberately left unchanged (skip this frame as if it
            # were never there) rather than set to None, so the NEXT
            # real frame still has a valid diff partner.
            speeds.append(speeds[-1] if speeds else 0.0)
            continue

        if prev_gray is None:
            speeds.append(0.0)  # placeholder, overwritten just below
            prev_gray = gray
            continue

        cache_path = None
        if cache:
            key = _frame_pair_cache_key(img_paths[i - 1], path)
            cache_path = os.path.join(SPEED_CACHE_DIR, key + '.pkl')
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    speeds.append(pickle.load(f))
                prev_gray = gray
                continue

        scaled_bbox = [v * scale for v in bbox]
        mag = _compute_flow_magnitude(prev_gray, gray, exclude_bbox=scaled_bbox)
        speeds.append(mag)
        if cache_path is not None:
            with open(cache_path, 'wb') as f:
                pickle.dump(mag, f)
        prev_gray = gray

    if len(speeds) > 1:
        speeds[0] = speeds[1]  # first frame: no prior frame to diff, reuse frame 2's value

    return [[s] for s in speeds]


def inject_synthetic_speed(data_dict, cache=True):
    """data_dict: the dict returned by JAAD.generate_data_trajectory_sequence
    (keys: image, pid, bbox, center, occlusion, intent, vehicle_act).
    Adds an 'obd_speed' key, same per-sample/per-frame structure as
    'bbox' (list of samples, each a list of per-frame values), computed
    via optical flow. Mutates and returns data_dict."""
    n_samples = len(data_dict['image'])
    obd_speed = []
    for i in range(n_samples):
        img_paths = data_dict['image'][i]
        bboxes = data_dict['bbox'][i]
        obd_speed.append(estimate_speed_for_track(img_paths, bboxes, cache=cache))
    data_dict['obd_speed'] = obd_speed
    return data_dict
