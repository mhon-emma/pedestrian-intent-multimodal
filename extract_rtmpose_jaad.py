"""
extract_rtmpose_jaad.py
========================
Extracts RTMPose keypoints for all JAAD pedestrians and saves a single pkl
compatible with the SF-GRU pose backend system.

Output format:
  pose_jaad_rtmpose.pkl  ->  {video_id: {key: [36 floats]}}
  key = f'{frame:05d}_{ped_id}'    (ped_id = JAAD track "new_id", e.g. '0_1_3b')
  36 floats = 18 OpenPose-style keypoints x (x/W, y/H) normalized to [0,1]

Keypoint order (SF-GRU OpenPose-18):
  [nose, neck, Rsho, Relb, Rwri, Lsho, Lelb, Lwri,
   Rhip, Rkne, Rank, Lhip, Lkne, Lank, Leye, Reye, Lear, Rear]

Usage:
  python extract_rtmpose_jaad.py [--update] [--device cuda]

The script is incremental: if the pkl already exists and --update is passed,
only videos not yet in the pkl are processed and merged in.

Unlike PIE, JAAD videos have varying resolutions (read from each video's XML),
and pedestrian ids come from the CVAT-style track attribute "id" (new_id) on
each <box>, not a per-frame "id" like PIE's ped_xml_id.
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

# -- Paths --------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
JAAD_DIR   = '/usr1/home/mehon/JAAD'
CLIPS_DIR  = os.path.join(JAAD_DIR, 'JAAD_clips')
ANNOT_DIR  = os.path.join(JAAD_DIR, 'annotations')
POSE_DIR   = os.path.join(BASE_DIR, 'data', 'features', 'jaad', 'poses')
OUT_PATH   = os.path.join(POSE_DIR, 'pose_jaad_rtmpose.pkl')

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


# COCO-17 -> SF-GRU OpenPose-18 (see extract_rtmpose.py for full mapping notes)
_COCO_TO_SFGRU = [0, None, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 1, 2, 3, 4]


def coco17_to_sfgru18(kpts: np.ndarray, img_w: int, img_h: int) -> list:
    neck = (kpts[5] + kpts[6]) / 2.0
    result = []
    for idx in _COCO_TO_SFGRU:
        kpt = neck if idx is None else kpts[idx]
        result.append(float(kpt[0]) / img_w)
        result.append(float(kpt[1]) / img_h)
    return result


def parse_annotations(annot_path: str):
    """
    Parse a JAAD CVAT-style XML file for pedestrian/people tracks.

    Returns:
        frame_peds: {frame_num: [(new_id, x1, y1, x2, y2), ...]}
        img_w, img_h: video resolution from the XML meta block
    """
    tree = ET.parse(annot_path)
    img_w = int(tree.find('./meta/task/original_size/width').text)
    img_h = int(tree.find('./meta/task/original_size/height').text)

    frame_peds = {}
    for track in tree.findall('./track'):
        label = track.get('label')
        if label not in ('pedestrian', 'ped', 'people'):
            continue

        boxes = track.findall('./box')
        if not boxes:
            continue

        id_attr = boxes[0].find('./attribute[@name="id"]')
        if id_attr is None or not id_attr.text:
            continue
        new_id = id_attr.text.strip()

        for box in boxes:
            if int(box.get('outside')) == 1:
                continue

            frame = int(box.get('frame'))
            x1 = max(0.0, float(box.get('xtl')))
            y1 = max(0.0, float(box.get('ytl')))
            x2 = min(float(img_w), float(box.get('xbr')))
            y2 = min(float(img_h), float(box.get('ybr')))

            if x2 <= x1 or y2 <= y1:
                continue

            frame_peds.setdefault(frame, []).append((new_id, x1, y1, x2, y2))

    return frame_peds, img_w, img_h


def extract_video(video_path, annot_path, pose_model):
    frame_peds, img_w, img_h = parse_annotations(annot_path)
    if not frame_peds:
        print('  No pedestrian annotations found.')
        return {}

    sorted_frames = sorted(frame_peds.keys())
    print(f'  {len(sorted_frames)} annotated frames, '
          f'{sum(len(v) for v in frame_peds.values())} ped-frame entries, '
          f'res={img_w}x{img_h}')

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
            continue

        peds = frame_peds[frame_num]

        if frame_num != prev_frame + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

        ret, frame_img = cap.read()
        prev_frame = frame_num if ret else -1

        if not ret:
            for new_id, *_ in peds:
                key = f'{frame_num:05d}_{new_id}'
                vid_poses[key] = [0.0] * 36
            continue

        bboxes = [[x1, y1, x2, y2] for (_, x1, y1, x2, y2) in peds]

        try:
            keypoints, scores = pose_model(frame_img, bboxes)
        except Exception as exc:
            print(f'  RTMPose error at frame {frame_num}: {exc}')
            for new_id, *_ in peds:
                key = f'{frame_num:05d}_{new_id}'
                vid_poses[key] = [0.0] * 36
            continue

        for i, (new_id, *_) in enumerate(peds):
            key = f'{frame_num:05d}_{new_id}'
            vid_poses[key] = coco17_to_sfgru18(keypoints[i], img_w, img_h)

        pbar.set_postfix(poses=len(vid_poses))

    pbar.close()
    cap.release()
    return vid_poses


def main():
    parser = argparse.ArgumentParser(description='Extract RTMPose keypoints for JAAD.')
    parser.add_argument('--update', action='store_true',
                        help='Merge new videos into existing pkl instead of reprocessing everything')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    args = parser.parse_args()

    pose_model = build_pose_model(device=args.device)

    existing = {}
    if args.update and os.path.isfile(OUT_PATH):
        with open(OUT_PATH, 'rb') as f:
            existing = pickle.load(f)
        print(f'Loaded existing pkl with {len(existing)} videos.')

    all_poses = dict(existing)
    new_videos = 0

    annot_files = sorted(f for f in os.listdir(ANNOT_DIR) if f.endswith('.xml'))
    for annot_file in annot_files:
        vid_id = annot_file.replace('.xml', '')  # e.g. video_0001

        if args.update and vid_id in existing:
            continue

        video_path = os.path.join(CLIPS_DIR, vid_id + '.mp4')
        if not os.path.isfile(video_path):
            print(f'[{vid_id}] No video file, skipping.')
            continue

        annot_path = os.path.join(ANNOT_DIR, annot_file)
        print(f'\n[{vid_id}] Processing...')

        vid_poses = extract_video(video_path, annot_path, pose_model)
        print(f'  -> {len(vid_poses)} pose entries')

        all_poses[vid_id] = vid_poses
        new_videos += 1

        os.makedirs(POSE_DIR, exist_ok=True)
        with open(OUT_PATH, 'wb') as f:
            pickle.dump(all_poses, f)
        print(f'  -> Saved checkpoint: {OUT_PATH}')

    if new_videos == 0:
        print('Nothing new to process.')
    else:
        print(f'\nDone. {new_videos} videos processed -> {OUT_PATH}')


if __name__ == '__main__':
    main()
