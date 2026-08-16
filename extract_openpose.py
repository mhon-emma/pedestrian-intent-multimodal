"""
extract_openpose.py
====================
Extracts OpenPose (COCO-18 body model) keypoints for all PIE pedestrians and
saves per-set pkl files compatible with the SF-GRU pose backend system.
Mirrors extract_rtmpose.py's structure and output format exactly, so both
plug into sf_gru_torch.py's get_pose lookup unchanged.

Output format:
  pose_{set_id}_openpose_full.pkl  ->  {video_id: {key: [36 floats]}}
  key = f'{frame:05d}_{set_num}_{vid_num}_{ped_xml_id}'
  36 floats = 18 keypoints x (x/W, y/H) normalized to [0,1]

  (named *_openpose_full.pkl, not *_openpose.pkl, so it never collides with
  the original authors' set03-only pose_set03.pkl cache -- see README.)

Keypoint order (SF-GRU OpenPose-18, COCO body model's native order --
no remapping needed, unlike RTMPose's COCO-17):
  [nose, neck, Rsho, Relb, Rwri, Lsho, Lelb, Lwri,
   Rhip, Rkne, Rank, Lhip, Lkne, Lank, Leye, Reye, Lear, Rear]

Unlike RTMPose (top-down: pose_model(frame, bboxes) gives keypoints already
matched to each bbox), OpenPose is bottom-up: it detects all people in a
frame at once with no bbox conditioning. This script runs OpenPose once per
video (its --video mode, avoiding per-frame process-startup cost), writes
one JSON per frame, then matches each detected skeleton to the correct
annotated pedestrian bbox by keypoint-derived bbox IoU.

Usage:
  python extract_openpose.py [--sets set01 set03 ...] [--update]

The script is incremental: if a pkl already exists and --update is passed,
only videos not yet in the pkl are processed and merged in.
"""

import argparse
import glob
import json
import os
import pickle
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

# -- Paths ----------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = '/usr1/home/mehon/PIE_clips'
ANNOT_DIR  = '/usr1/home/mehon/PIE/annotations/annotations'
POSE_DIR   = os.path.join(BASE_DIR, 'data', 'features', 'pie', 'poses')

OPENPOSE_BIN    = '/usr1/home/mehon/openpose/build/examples/openpose/openpose.bin'
OPENPOSE_MODELS = '/usr1/home/mehon/openpose/models/'
OPENPOSE_LIBS   = (
    '/usr1/home/mehon/openpose/build/caffe/lib:'
    '/usr1/home/mehon/openpose/build/src/openpose'
)

IMG_W = 1920
IMG_H = 1080

# COCO-18 body model's native keypoint order already matches SF-GRU's
# expected order exactly -- no remapping needed (contrast with RTMPose's
# COCO-17, which needed _COCO_TO_SFGRU reindexing in extract_rtmpose.py).
NUM_KEYPOINTS = 18


def run_openpose_on_video(video_path: str, out_dir: str) -> None:
    """Run the OpenPose binary once on a full video, writing one
    <frame>_keypoints.json per frame into out_dir."""
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = OPENPOSE_LIBS + ':' + env.get('LD_LIBRARY_PATH', '')
    cmd = [
        OPENPOSE_BIN,
        '--video', video_path,
        '--model_pose', 'COCO',
        '--model_folder', OPENPOSE_MODELS,
        '--write_json', out_dir,
        '--display', '0',
        '--render_pose', '0',
    ]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3600)
    if result.returncode != 0:
        raise RuntimeError(
            'OpenPose failed (exit %d) on %s:\nstdout: %s\nstderr: %s'
            % (result.returncode, video_path, result.stdout[-2000:], result.stderr[-2000:])
        )


def _bbox_from_keypoints(kpts_flat):
    """kpts_flat: flat [x0,y0,c0,x1,y1,c1,...] list from OpenPose JSON.
    Returns (x1,y1,x2,y2) of valid (nonzero-confidence) points, or None."""
    xs = kpts_flat[0::3]
    ys = kpts_flat[1::3]
    cs = kpts_flat[2::3]
    valid = [(x, y) for x, y, c in zip(xs, ys, cs) if c > 0]
    if not valid:
        return None
    vx = [v[0] for v in valid]
    vy = [v[1] for v in valid]
    return (min(vx), min(vy), max(vx), max(vy))


def _iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_people_to_bboxes(people, ped_bboxes, iou_threshold=0.1):
    """
    people: list of OpenPose detections (each a dict with 'pose_keypoints_2d')
    ped_bboxes: list of (ped_xml_id, x1, y1, x2, y2) annotated pedestrian boxes
    Returns: {ped_xml_id: pose_keypoints_2d (flat list)} for matched pairs.
    Greedy highest-IoU matching; one detection can match at most one bbox.
    """
    det_boxes = []
    for p in people:
        b = _bbox_from_keypoints(p['pose_keypoints_2d'])
        det_boxes.append(b)

    pairs = []
    for pi, (ped_id, *ann_box) in enumerate(ped_bboxes):
        for di, det_box in enumerate(det_boxes):
            if det_box is None:
                continue
            score = _iou(tuple(ann_box), det_box)
            if score >= iou_threshold:
                pairs.append((score, pi, di))
    pairs.sort(reverse=True)

    matched_ped = set()
    matched_det = set()
    result = {}
    for score, pi, di in pairs:
        if pi in matched_ped or di in matched_det:
            continue
        matched_ped.add(pi)
        matched_det.add(di)
        ped_id = ped_bboxes[pi][0]
        result[ped_id] = people[di]['pose_keypoints_2d']
    return result


def openpose_kpts_to_sfgru18(kpts_flat) -> list:
    """kpts_flat: 18*3 flat [x,y,c,...] in COCO-18 order (already matches
    SF-GRU order). Returns 36 floats [x0/W,y0/H,...], zeroed where c==0."""
    result = []
    for i in range(NUM_KEYPOINTS):
        x, y, c = kpts_flat[i * 3], kpts_flat[i * 3 + 1], kpts_flat[i * 3 + 2]
        if c <= 0:
            result.append(0.0)
            result.append(0.0)
        else:
            result.append(float(x) / IMG_W)
            result.append(float(y) / IMG_H)
    return result


def parse_annotations(annot_path: str) -> dict:
    """Same CVAT XML parser as extract_rtmpose.py."""
    tree = ET.parse(annot_path)
    frame_peds: dict = {}
    for track in tree.findall('./track'):
        if track.get('label') != 'pedestrian':
            continue
        ped_xml_id = None
        for box in track.findall('./box'):
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
            x1 = max(0.0, float(box.get('xtl')))
            y1 = max(0.0, float(box.get('ytl')))
            x2 = min(float(IMG_W), float(box.get('xbr')))
            y2 = min(float(IMG_H), float(box.get('ybr')))
            if x2 <= x1 or y2 <= y1:
                continue
            frame_peds.setdefault(frame, []).append((ped_xml_id, x1, y1, x2, y2))
    return frame_peds


def extract_video(video_path: str, annot_path: str, set_num: int, vid_num: str) -> dict:
    """
    Runs OpenPose on the whole video once, then matches detections to
    annotated pedestrian bboxes per frame.
    Returns: {key: [36 floats]} where key = f'{frame:05d}_{set_num}_{vid_num}_{ped_id}'
    """
    frame_peds = parse_annotations(annot_path)
    if not frame_peds:
        print('  No pedestrian annotations found.')
        return {}

    print(f'  {len(frame_peds)} annotated frames, '
          f'{sum(len(v) for v in frame_peds.values())} ped-frame entries')

    tmp_dir = tempfile.mkdtemp(prefix='openpose_json_')
    try:
        print('  Running OpenPose on full video...')
        run_openpose_on_video(video_path, tmp_dir)

        vid_poses = {}
        matched_frames = 0
        for frame_num, peds in frame_peds.items():
            json_path = os.path.join(
                tmp_dir, '%s_%012d_keypoints.json' % (os.path.splitext(os.path.basename(video_path))[0], frame_num)
            )
            if not os.path.isfile(json_path):
                for ped_xml_id, *_ in peds:
                    key = f'{frame_num:05d}_{set_num}_{vid_num}_{ped_xml_id}'
                    vid_poses[key] = [0.0] * 36
                continue

            with open(json_path, 'r') as f:
                data = json.load(f)
            people = data.get('people', [])

            matches = match_people_to_bboxes(people, peds)
            for ped_xml_id, *_ in peds:
                key = f'{frame_num:05d}_{set_num}_{vid_num}_{ped_xml_id}'
                if ped_xml_id in matches:
                    vid_poses[key] = openpose_kpts_to_sfgru18(matches[ped_xml_id])
                    matched_frames += 1
                else:
                    vid_poses[key] = [0.0] * 36

        total = sum(len(v) for v in frame_peds.values())
        print(f'  Matched {matched_frames}/{total} ped-frame entries to OpenPose detections')
        return vid_poses
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def process_set(set_id: str, update: bool = False) -> None:
    videos_dir = os.path.join(DATA_DIR, set_id)
    annot_dir  = os.path.join(ANNOT_DIR, set_id)
    out_path   = os.path.join(POSE_DIR, f'pose_{set_id}_openpose_full.pkl')

    if not os.path.isdir(annot_dir):
        print(f'[{set_id}] No annotations directory, skipping.')
        return
    if not os.path.isdir(videos_dir):
        print(f'[{set_id}] No videos directory, skipping.')
        return

    set_num = int(set_id.replace('set', ''))

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
        vid_id  = annot_file.replace('_annt.xml', '')
        vid_num = vid_id.replace('video_', '')

        if update and vid_id in existing_poses:
            print(f'[{set_id}/{vid_id}] Already processed, skipping.')
            continue

        video_path = os.path.join(videos_dir, vid_id + '.mp4')
        if not os.path.isfile(video_path):
            print(f'[{set_id}/{vid_id}] No video file yet, skipping.')
            continue

        annot_path = os.path.join(annot_dir, annot_file)
        print(f'\n[{set_id}/{vid_id}] Processing...')

        vid_poses = extract_video(video_path, annot_path, set_num, vid_num)
        print(f'  -> {len(vid_poses)} pose entries')

        set_poses[vid_id] = vid_poses
        new_videos += 1

        os.makedirs(POSE_DIR, exist_ok=True)
        with open(out_path, 'wb') as f:
            pickle.dump(set_poses, f)
        print(f'  -> Saved checkpoint: {out_path}')

    if new_videos == 0:
        print(f'[{set_id}] Nothing new to process.')
    else:
        print(f'\n[{set_id}] Done. {new_videos} videos processed -> {out_path}')


def main():
    parser = argparse.ArgumentParser(description='Extract OpenPose keypoints for all PIE sets.')
    parser.add_argument('--sets', nargs='+',
                        default=['set01', 'set02', 'set03', 'set04', 'set05', 'set06'],
                        help='Which sets to process (default: all six)')
    parser.add_argument('--update', action='store_true',
                        help='Merge new videos into existing pkl instead of reprocessing everything')
    args = parser.parse_args()

    if not os.path.isfile(OPENPOSE_BIN):
        raise SystemExit('OpenPose binary not found at %s -- build it first.' % OPENPOSE_BIN)

    for set_id in args.sets:
        print(f'\n{"="*60}')
        print(f'SET: {set_id}')
        print('='*60)
        process_set(set_id, update=args.update)

    print('\nAll done.')


if __name__ == '__main__':
    main()
