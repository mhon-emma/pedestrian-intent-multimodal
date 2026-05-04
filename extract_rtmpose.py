"""
extract_rtmpose.py
==================
Extracts RTMPose keypoints for all PIE pedestrians and saves per-set pkl files
compatible with the SF-GRU pose backend system.

Output format:
  pose_{set_id}_rtmpose.pkl  →  {video_id: {key: [36 floats]}}
  key = f'{frame:05d}_{set_num}_{vid_num}_{ped_xml_id}'
  36 floats = 18 OpenPose-style keypoints × (x/W, y/H) normalized to [0,1]

Keypoint order (SF-GRU OpenPose-18):
  [nose, neck, Rsho, Relb, Rwri, Lsho, Lelb, Lwri,
   Rhip, Rkne, Rank, Lhip, Lkne, Lank, Leye, Reye, Lear, Rear]

Usage:
  conda run -n mmml python extract_rtmpose.py [--sets set01 set03 ...]
                                               [--update]   # merge into existing pkl

The script is incremental: if a pkl already exists and --update is passed,
only videos not yet in the pkl are processed and merged in.
"""

import argparse
import os
import pickle
import site
import sys
import xml.etree.ElementTree as ET


def _setup_nvidia_libs():
    """Prepend all pip-installed nvidia CUDA libs to LD_LIBRARY_PATH."""
    lib_paths = []
    for p in site.getsitepackages():
        nvidia_dir = os.path.join(p, 'nvidia')
        if not os.path.isdir(nvidia_dir):
            continue
        for sub in sorted(os.listdir(nvidia_dir)):
            lib = os.path.join(nvidia_dir, sub, 'lib')
            if os.path.isdir(lib):
                lib_paths.append(lib)
    if lib_paths:
        extra = ':'.join(lib_paths)
        os.environ['LD_LIBRARY_PATH'] = extra + ':' + os.environ.get('LD_LIBRARY_PATH', '')


_setup_nvidia_libs()

import cv2
import numpy as np
from rtmlib import RTMPose
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = '/home/teamj/Documents/MMML/Data'
ANNOT_DIR = '/home/teamj/Documents/MMML/PIE/annotations/annotations'
POSE_DIR  = os.path.join(BASE_DIR, 'data', 'features', 'pie', 'poses')

IMG_W = 1920
IMG_H = 1080

# ── RTMPose model (downloaded on first run to ~/.cache/rtmlib/checkpoints) ────
MODEL_URL = (
    'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/'
    'rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip'
)


def build_pose_model(device='cuda'):
    print(f'Loading RTMPose model on {device}...')
    return RTMPose(
        onnx_model=MODEL_URL,
        model_input_size=(192, 256),
        backend='onnxruntime',
        device=device,
    )


# ── COCO-17 → SF-GRU OpenPose-18 conversion ───────────────────────────────────
# COCO-17 indices:
#   0=nose, 1=Leye, 2=Reye, 3=Lear, 4=Rear,
#   5=Lsho, 6=Rsho, 7=Lelb, 8=Relb, 9=Lwri, 10=Rwri,
#   11=Lhip, 12=Rhip, 13=Lkne, 14=Rkne, 15=Lank, 16=Rank
#
# SF-GRU order: [nose, neck, Rsho, Relb, Rwri, Lsho, Lelb, Lwri,
#                Rhip, Rkne, Rank, Lhip, Lkne, Lank, Leye, Reye, Lear, Rear]
#  None → compute neck as midpoint of Lsho(5) and Rsho(6)
_COCO_TO_SFGRU = [0, None, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 1, 2, 3, 4]


def coco17_to_sfgru18(kpts: np.ndarray) -> list:
    """
    kpts: (17, 2) absolute pixel coordinates from RTMPose
    Returns: list of 36 floats [x0/W, y0/H, x1/W, y1/H, ...]
    """
    neck = (kpts[5] + kpts[6]) / 2.0
    result = []
    for idx in _COCO_TO_SFGRU:
        kpt = neck if idx is None else kpts[idx]
        result.append(float(kpt[0]) / IMG_W)
        result.append(float(kpt[1]) / IMG_H)
    return result


# ── Annotation parser ──────────────────────────────────────────────────────────

def parse_annotations(annot_path: str) -> dict:
    """
    Parse a CVAT XML annotation file for pedestrian tracks.

    Returns:
        {frame_num: [(ped_xml_id, x1, y1, x2, y2), ...]}
    """
    tree = ET.parse(annot_path)
    frame_peds: dict[int, list] = {}

    for track in tree.findall('./track'):
        if track.get('label') != 'pedestrian':
            continue

        ped_xml_id = None
        for box in track.findall('./box'):
            # Extract pedestrian id from attributes (read once per track)
            if ped_xml_id is None:
                for attr in box.findall('./attribute'):
                    if attr.get('name') == 'id':
                        ped_xml_id = attr.text.strip()
                        break
            if ped_xml_id is None:
                continue

            if int(box.get('outside')) == 1:
                continue

            frame = int(box.get('frame'))
            x1 = float(box.get('xtl'))
            y1 = float(box.get('ytl'))
            x2 = float(box.get('xbr'))
            y2 = float(box.get('ybr'))

            # Clamp to image bounds
            x1 = max(0.0, x1); y1 = max(0.0, y1)
            x2 = min(float(IMG_W), x2); y2 = min(float(IMG_H), y2)

            if x2 <= x1 or y2 <= y1:
                continue

            frame_peds.setdefault(frame, []).append((ped_xml_id, x1, y1, x2, y2))

    return frame_peds


# ── Per-video extraction ───────────────────────────────────────────────────────

def extract_video(
    video_path: str,
    annot_path: str,
    set_num: int,
    vid_num: str,
    pose_model: RTMPose,
) -> dict:
    """
    Extract RTMPose keypoints for all annotated pedestrians in one video.

    Returns:
        {key: [36 floats]}  where key = f'{frame:05d}_{set_num}_{vid_num}_{ped_id}'
    """
    frame_peds = parse_annotations(annot_path)
    if not frame_peds:
        print(f'  No pedestrian annotations found.')
        return {}

    sorted_frames = sorted(frame_peds.keys())
    print(f'  {len(sorted_frames)} annotated frames, '
          f'{sum(len(v) for v in frame_peds.values())} ped-frame entries')

    vid_poses = {}
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f'  ERROR: Cannot open video {video_path}')
        return {}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    prev_frame = -1

    pbar = tqdm(sorted_frames, unit='frame', dynamic_ncols=True)
    for frame_num in pbar:
        if frame_num >= total_frames:
            print(f'  WARNING: frame {frame_num} >= total {total_frames}, skipping')
            continue

        peds = frame_peds[frame_num]

        # Seek only when frames are non-consecutive
        if frame_num != prev_frame + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

        ret, frame_img = cap.read()
        prev_frame = frame_num if ret else -1

        if not ret:
            print(f'  WARNING: failed to read frame {frame_num}')
            for ped_xml_id, *_ in peds:
                key = f'{frame_num:05d}_{set_num}_{vid_num}_{ped_xml_id}'
                vid_poses[key] = [0.0] * 36
            continue

        bboxes = [[x1, y1, x2, y2] for (_, x1, y1, x2, y2) in peds]

        try:
            keypoints, scores = pose_model(frame_img, bboxes)
            # keypoints: (N, 17, 2) absolute pixel coords
        except Exception as exc:
            print(f'  RTMPose error at frame {frame_num}: {exc}')
            for ped_xml_id, *_ in peds:
                key = f'{frame_num:05d}_{set_num}_{vid_num}_{ped_xml_id}'
                vid_poses[key] = [0.0] * 36
            continue

        for i, (ped_xml_id, *_) in enumerate(peds):
            key = f'{frame_num:05d}_{set_num}_{vid_num}_{ped_xml_id}'
            vid_poses[key] = coco17_to_sfgru18(keypoints[i])

        pbar.set_postfix(poses=len(vid_poses))

    pbar.close()
    cap.release()
    return vid_poses


# ── Per-set extraction ─────────────────────────────────────────────────────────

def process_set(set_id: str, pose_model: RTMPose, update: bool = False) -> None:
    videos_dir = os.path.join(DATA_DIR, set_id)
    annot_dir  = os.path.join(ANNOT_DIR, set_id)
    out_path   = os.path.join(POSE_DIR, f'pose_{set_id}_rtmpose.pkl')

    if not os.path.isdir(annot_dir):
        print(f'[{set_id}] No annotations directory, skipping.')
        return

    if not os.path.isdir(videos_dir):
        print(f'[{set_id}] No videos directory, skipping.')
        return

    set_num = int(set_id.replace('set', ''))

    # Load existing pkl if updating incrementally
    existing_poses: dict = {}
    if update and os.path.isfile(out_path):
        with open(out_path, 'rb') as f:
            existing_poses = pickle.load(f)
        print(f'[{set_id}] Loaded existing pkl with {len(existing_poses)} videos.')

    set_poses = dict(existing_poses)
    new_videos = 0

    for annot_file in sorted(os.listdir(annot_dir)):
        if not annot_file.endswith('_annt.xml'):
            continue

        vid_id  = annot_file.replace('_annt.xml', '')   # e.g. video_0001
        vid_num = vid_id.replace('video_', '')           # e.g. 0001

        if update and vid_id in existing_poses:
            print(f'[{set_id}/{vid_id}] Already processed, skipping.')
            continue

        video_path = os.path.join(videos_dir, vid_id + '.mp4')
        if not os.path.isfile(video_path):
            print(f'[{set_id}/{vid_id}] No video file yet, skipping.')
            continue

        annot_path = os.path.join(annot_dir, annot_file)
        print(f'\n[{set_id}/{vid_id}] Processing...')

        vid_poses = extract_video(
            video_path, annot_path, set_num, vid_num, pose_model
        )
        print(f'  -> {len(vid_poses)} pose entries')

        set_poses[vid_id] = vid_poses
        new_videos += 1

        # Save after each video so progress isn't lost on interruption
        os.makedirs(POSE_DIR, exist_ok=True)
        with open(out_path, 'wb') as f:
            pickle.dump(set_poses, f)
        print(f'  -> Saved checkpoint: {out_path}')

    if new_videos == 0:
        print(f'[{set_id}] Nothing new to process.')
    else:
        print(f'\n[{set_id}] Done. {new_videos} videos processed → {out_path}')


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Extract RTMPose keypoints for all PIE sets.'
    )
    parser.add_argument(
        '--sets', nargs='+',
        default=['set01', 'set02', 'set03', 'set04', 'set05', 'set06'],
        help='Which sets to process (default: all six)',
    )
    parser.add_argument(
        '--update', action='store_true',
        help='Merge new videos into existing pkl instead of reprocessing everything',
    )
    parser.add_argument(
        '--device', default='cuda', choices=['cuda', 'cpu'],
        help='Device for RTMPose inference (default: cuda)',
    )
    args = parser.parse_args()

    pose_model = build_pose_model(device=args.device)

    for set_id in args.sets:
        print(f'\n{"="*60}')
        print(f'SET: {set_id}')
        print('='*60)
        process_set(set_id, pose_model, update=args.update)

    print('\nAll done.')


if __name__ == '__main__':
    main()
