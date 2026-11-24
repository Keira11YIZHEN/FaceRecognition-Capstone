r"""NTU Capstone progress demo: detection, identity, tracking and lip motion.

Run inside FaceRecognition_Capstone:
    .\.venv\Scripts\python.exe demo_showcase.py

Required neighboring files: face_demo.py, combined_demo.py
Required model: models/face_landmarker.task, plus the two face .onnx models.
Controls: Q quit, S save a screenshot to captures/.

This is a LOCAL VISUAL prototype. It does not measure 1.5 m distance,
identify active speech, authenticate against spoofing, or control a robot.
"""

from collections import deque
from datetime import datetime
from pathlib import Path
import time

import cv2
import mediapipe as mp
import numpy as np
import supervision as sv

import combined_demo as core  # Reuse working face + lip helper functions

fd = core.fd
ROOT = Path(__file__).resolve().parent
MODEL = ROOT / 'models' / 'face_landmarker.task'
OUT_DIR = ROOT / 'captures'

# First-pass demo parameters, NOT validated biometric/speech thresholds.
CONFIRM_FRAMES = 3
HISTORY_LENGTH = 12
MOTION_THRESHOLD = 0.045
MAX_FACES = 4
MIN_FACE_IOU = 0.35
MIN_LIP_IOU = 0.12

DARK = (34, 28, 22)      # OpenCV uses BGR colors
CARD = (48, 42, 37)
WHITE = (242, 242, 242)
MUTED = (177, 173, 165)
GREEN = (117, 220, 108)
YELLOW = (112, 211, 253)
RED = (112, 111, 242)
BLUE = (240, 181, 110)

PANEL_W = 340
VIDEO_W = 870


def txt(canvas, text, x, y, scale=0.54, color=WHITE, thickness=1):
    cv2.putText(canvas, str(text), (int(x), int(y)),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def info_line(panel, y, label, value, value_color=WHITE):
    txt(panel, label.upper(), 24, y, 0.42, MUTED)
    txt(panel, value, 24, y + 25, 0.70, value_color, 2)
    cv2.line(panel, (22, y + 39), (PANEL_W - 22, y + 39), CARD, 1)


def panel_from_state(height, data):
    panel = np.full((height, PANEL_W, 3), DARK, dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (PANEL_W, 63), CARD, -1)
    txt(panel, 'CAPSTONE  /  LIVE DEMO', 20, 28, 0.63, WHITE, 2)
    txt(panel, 'Face + Identity + Tracking + Lips', 20, 50, 0.43, MUTED)

    # Proportionate spacing so the dashboard fits normal laptop webcams.
    top = 82
    spacing = max(57, min(72, (height - 146) // 6))
    entries = [
        ('Face detected', 'YES' if data['faces'] else 'NO', GREEN if data['faces'] else MUTED),
        ('Identity', data['identity'], GREEN if data['identity'] not in ('UNKNOWN', 'VERIFYING', 'NONE') else YELLOW if data['identity'] == 'VERIFYING' else RED if data['identity'] == 'UNKNOWN' else MUTED),
        ('Track ID', data['track'], BLUE if data['track'] != '--' else MUTED),
        ('Lip motion', data['lip'], YELLOW if data['lip'] == 'MOVING' else GREEN if data['lip'] == 'STILL' else MUTED),
        ('FPS / processing', f"{data['fps']:.1f} FPS  /  {data['latency']:.0f} ms", WHITE),
        ('Faces in frame', str(data['faces']), WHITE),
    ]
    for i, (label, value, color) in enumerate(entries):
        info_line(panel, top + spacing * i, label, value, color)

    cv2.rectangle(panel, (16, height - 69), (PANEL_W - 16, height - 13), CARD, -1)
    txt(panel, 'DISTANCE: NOT MEASURED', 25, height - 48, 0.44, YELLOW)
    txt(panel, 'SPEECH: NOT VERIFIED', 25, height - 28, 0.44, YELLOW)
    return panel


def paint_face(frame, box, name, tid, lip_status, confidence=None):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    recognized = name not in ('UNKNOWN', 'VERIFYING')
    color = GREEN if recognized else YELLOW if name == 'VERIFYING' else RED
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    if confidence is not None and recognized:
        label = f'{name}  |  Track {tid}  |  {lip_status}  |  Sim {confidence:.2f}'
    else:
        label = f'{name}  |  Track {tid}'
    label_scale = 0.53
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, label_scale, 2)
    tx = max(4, min(x1, width - text_w - 8))
    ty = y1 - 9 if y1 > 32 else min(height - 12, y2 + 22)
    cv2.rectangle(frame, (tx - 3, ty - text_h - 5),
                  (min(width - 1, tx + text_w + 5), ty + 5), DARK, -1)
    txt(frame, label, tx, ty, label_scale, color, 2)


def main():
    if not MODEL.is_file():
        raise FileNotFoundError(f'Please put face_landmarker.task in models/: {MODEL}')
    if not hasattr(sv, 'ByteTrack'):
        raise RuntimeError('ByteTrack missing. Install: pip install supervision==0.25.0')
    profiles = fd.load_profiles()
    if not profiles:
        raise RuntimeError('No registered users. First run face_demo.py enroll user1')

    tracker = sv.ByteTrack(track_activation_threshold=0.25,
                           lost_track_buffer=15, frame_rate=30,
                           minimum_consecutive_frames=1)
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=MAX_FACES,
    )

    cap = fd.open_camera()
    confirmations = {}  # tid -> (name, consecutive count)
    lip_histories = {}  # tid -> (name, history)
    last_seen = {}
    frame_number = 0
    last_time = time.perf_counter()
    smoothed_fps = 0.0

    print('CAPSTONE SHOWCASE: Q=quit, S=save screenshot')
    print('Warning: no distance measurement, audio speaker detection, or robot control.')
    try:
        with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print('Could not read camera frame.')
                    break
                frame_number += 1
                height, width = frame.shape[:2]
                start = time.perf_counter()

                faces = fd.find_faces(frame)
                boxes, detections = core.detections_from_faces(faces)
                tracks = tracker.update_with_detections(detections)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                    data=np.ascontiguousarray(rgb))
                result = landmarker.detect(mp_image)
                landmark_faces = result.face_landmarks or []
                landmark_boxes = [core.landmarks_box(lm, width, height)
                                  for lm in landmark_faces]
                taken_landmarks = set()
                view = frame.copy()
                featured = None  # (priority, area, data)

                if tracks.tracker_id is not None:
                    for box, raw_tid in zip(tracks.xyxy, tracks.tracker_id):
                        if raw_tid is None or not len(faces):
                            continue
                        tid = int(raw_tid)
                        last_seen[tid] = frame_number
                        overlaps = [fd.iou(box, detected) for detected in boxes]
                        detected_idx = int(np.argmax(overlaps))
                        if overlaps[detected_idx] < MIN_FACE_IOU:
                            continue

                        name, similarity = fd.recognize(frame, faces[detected_idx], profiles)
                        old_name, old_count = confirmations.get(tid, ('Unknown', 0))
                        count = old_count + 1 if old_name == name else 1
                        confirmations[tid] = (name, count)
                        confirmed = name != 'Unknown' and count >= CONFIRM_FRAMES

                        identity = name if confirmed else 'UNKNOWN' if name == 'Unknown' else 'VERIFYING'
                        lip_state = 'NOT ANALYZED'
                        if confirmed:
                            # Match landmarks to this tracked/recognized face via face-box overlap.
                            best_idx, best_iou = None, 0.0
                            for idx, landmark_box in enumerate(landmark_boxes):
                                if idx in taken_landmarks:
                                    continue
                                score = fd.iou(box, landmark_box)
                                if score > best_iou:
                                    best_idx, best_iou = idx, score
                            if best_idx is not None and best_iou >= MIN_LIP_IOU:
                                taken_landmarks.add(best_idx)
                                landmarks = landmark_faces[best_idx]
                                ratio = core.mouth_opening_ratio(landmarks, width, height)
                                old_lip_name, history = lip_histories.get(
                                    tid, (name, deque(maxlen=HISTORY_LENGTH)))
                                if old_lip_name != name:
                                    history = deque(maxlen=HISTORY_LENGTH)
                                history.append(ratio)
                                lip_histories[tid] = (name, history)
                                moving = len(history) >= 8 and max(history) - min(history) > MOTION_THRESHOLD
                                lip_state = 'MOVING' if moving else 'STILL' if len(history) >= 8 else 'CALIBRATING'
                                core.draw_lips(view, landmarks, moving, width, height)
                            else:
                                lip_histories.pop(tid, None)
                                lip_state = 'NO LANDMARKS'
                        else:
                            lip_histories.pop(tid, None)

                        paint_face(view, box, identity, tid, lip_state, similarity)
                        area = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
                        # Prefer a confirmed registered user; otherwise display an unknown/verification status.
                        priority = 2 if confirmed else 1
                        candidate = {'identity': identity, 'track': str(tid), 'lip': lip_state}
                        if featured is None or (priority, area) > featured[:2]:
                            featured = (priority, area, candidate)

                for tid in list(last_seen):
                    if frame_number - last_seen[tid] > 45:
                        last_seen.pop(tid, None)
                        confirmations.pop(tid, None)
                        lip_histories.pop(tid, None)

                end = time.perf_counter()
                instantaneous_fps = 1.0 / max(end - last_time, 1e-6)
                smoothed_fps = (0.80 * smoothed_fps + 0.20 * instantaneous_fps
                                if smoothed_fps else instantaneous_fps)
                last_time = end
                latency_ms = (end - start) * 1000

                state = {'faces': len(faces), 'identity': 'NONE',
                         'track': '--', 'lip': 'WAITING',
                         'fps': smoothed_fps, 'latency': latency_ms}
                if featured is not None:
                    state.update(featured[2])

                # Render video, top bar, and the right status dashboard.
                display_h = int(round(height * (VIDEO_W / width)))
                display_h = max(display_h, 490)
                video = cv2.resize(view, (VIDEO_W, display_h),
                                   interpolation=cv2.INTER_LINEAR)
                cv2.rectangle(video, (0, 0), (VIDEO_W, 40), DARK, -1)
                txt(video, 'LIVE CAMERA  |  YuNet + SFace + ByteTrack + MediaPipe',
                    14, 26, 0.56, WHITE, 2)
                cv2.rectangle(video, (0, display_h - 32), (VIDEO_W, display_h), DARK, -1)
                txt(video, 'Q: quit     S: save screenshot     Prototype - visual lip motion only',
                    14, display_h - 10, 0.46, YELLOW)
                dashboard = panel_from_state(display_h, state)
                screen = np.hstack([video, dashboard])

                cv2.imshow('NTU Capstone  |  Progress Demo', screen)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
                if key == ord('s'):
                    OUT_DIR.mkdir(exist_ok=True)
                    path = OUT_DIR / f"capstone_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    if cv2.imwrite(str(path), screen):
                        print(f'Screenshot saved: {path}')
                    else:
                        print(f'Could not save screenshot: {path}')
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()