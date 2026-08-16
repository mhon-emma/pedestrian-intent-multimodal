"""
extract_openpose_jaad.py
==========================
Extracts OpenPose (COCO-18 body model) keypoints for all JAAD pedestrians and
saves a single pkl compatible with the SF-GRU pose backend system. Mirrors
extract_rtmpose_jaad.py's structure and output format exactly, and shares
extract_openpose.py's OpenPose-invocation / bbox-matching logic (see that
file for why OpenPose needs a match-detections-to-bboxes step, unlike
RTMPose's bbox-conditioned top-down inference).

Output format:
  pose_jaad_openpose_full.pkl  ->  {video_id: {key: [36 floats]}}
  key = f'{frame:05d}_{new_id}'   (new_id = JAAD track id, e.g. '0_1_3b')
  36 floats = 18 keypoints x (x/W, y/H) normalized to [0,1]

Usage:
  python extract_openpose_jaad.py [--update]

The script is incremental: if the pkl already exists and --update is
passed, only videos not yet in the pkl are processed and merged in.
"""

import argparse
import json
import os
import pickle
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET

from extract_openpose import (OPENPOSE_BIN, OPENPOSE_LIBS, OPENPOSE_MODELS,
                              match_people_to_bboxes,
                              openpose_kpts_to_sfgru18,
                              run_openpose_on_video)

# -- Paths ----------------------------------------------------------------------
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
JAAD_DIR  = '/usr1/home/mehon/JAAD'
CLIPS_DIR = os.path.join(JAAD_DIR, 'JAAD_clips')
ANNOT_DIR = os.path.join(JAAD_DIR, 'annotations')
POSE_DIR  = os.path.join(BASE_DIR, 'data', 'features', 'jaad', 'poses')
OUT_PATH  = os.path.join(POSE_DIR, 'pose_jaad_openpose_full.pkl')


def parse_annotations(annot_path: str):
    """Same JAAD CVAT-style XML parser as extract_rtmpose_jaad.py -- reads
    per-video resolution from the XML (JAAD videos aren't fixed-size like
    PIE) and uses the 'id' track attribute as new_id."""
    tree = ET.parse(annot_path)
    img_w = int(tree.find('./meta/task/original_size/width').text)
    img_h = int(tree.find('./meta/task/original_size/height').text)

    frame_peds = {}
    for track in tree.findall('./track'):
        if track.get('label') not in ('pedestrian', 'ped', 'people'):
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


def extract_video(video_path: str, annot_path: str) -> dict:
    frame_peds, img_w, img_h = parse_annotations(annot_path)
    if not frame_peds:
        print('  No pedestrian annotations found.')
        return {}

    print(f'  {len(frame_peds)} annotated frames, '
          f'{sum(len(v) for v in frame_peds.values())} ped-frame entries, '
          f'res={img_w}x{img_h}')

    tmp_dir = tempfile.mkdtemp(prefix='openpose_jaad_json_')
    try:
        print('  Running OpenPose on full video...')
        run_openpose_on_video(video_path, tmp_dir)

        vid_poses = {}
        matched = 0
        vid_base = os.path.splitext(os.path.basename(video_path))[0]
        for frame_num, peds in frame_peds.items():
            json_path = os.path.join(tmp_dir, '%s_%012d_keypoints.json' % (vid_base, frame_num))
            if not os.path.isfile(json_path):
                for new_id, *_ in peds:
                    vid_poses[f'{frame_num:05d}_{new_id}'] = [0.0] * 36
                continue

            with open(json_path, 'r') as f:
                data = json.load(f)
            people = data.get('people', [])

            matches = match_people_to_bboxes(people, peds)
            for new_id, *_ in peds:
                key = f'{frame_num:05d}_{new_id}'
                if new_id in matches:
                    # openpose_kpts_to_sfgru18 divides by module-level IMG_W/IMG_H
                    # (fixed 1920x1080 for PIE); JAAD resolution varies per video,
                    # so normalize here directly instead.
                    kpts_flat = matches[new_id]
                    result = []
                    for i in range(18):
                        x, y, c = kpts_flat[i*3], kpts_flat[i*3+1], kpts_flat[i*3+2]
                        if c <= 0:
                            result.extend([0.0, 0.0])
                        else:
                            result.extend([float(x) / img_w, float(y) / img_h])
                    vid_poses[key] = result
                    matched += 1
                else:
                    vid_poses[key] = [0.0] * 36

        total = sum(len(v) for v in frame_peds.values())
        print(f'  Matched {matched}/{total} ped-frame entries to OpenPose detections')
        return vid_poses
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description='Extract OpenPose keypoints for JAAD.')
    parser.add_argument('--update', action='store_true',
                        help='Merge new videos into existing pkl instead of reprocessing everything')
    args = parser.parse_args()

    if not os.path.isfile(OPENPOSE_BIN):
        raise SystemExit('OpenPose binary not found at %s -- build it first.' % OPENPOSE_BIN)

    existing = {}
    if args.update and os.path.isfile(OUT_PATH):
        with open(OUT_PATH, 'rb') as f:
            existing = pickle.load(f)
        print(f'Loaded existing pkl with {len(existing)} videos.')

    all_poses = dict(existing)
    new_videos = 0

    annot_files = sorted(f for f in os.listdir(ANNOT_DIR) if f.endswith('.xml'))
    for annot_file in annot_files:
        vid_id = annot_file.replace('.xml', '')

        if args.update and vid_id in existing:
            continue

        video_path = os.path.join(CLIPS_DIR, vid_id + '.mp4')
        if not os.path.isfile(video_path):
            print(f'[{vid_id}] No video file, skipping.')
            continue

        annot_path = os.path.join(ANNOT_DIR, annot_file)
        print(f'\n[{vid_id}] Processing...')

        vid_poses = extract_video(video_path, annot_path)
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
