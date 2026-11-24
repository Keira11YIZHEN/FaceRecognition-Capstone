r"""Combined NTU Capstone webcam prototype.

Single camera: YuNet detection -> SFace recognition -> ByteTrack tracking
-> MediaPipe Face Landmarker lip outline and mouth-motion estimation.

Keep face_demo.py alongside this file. Both scripts share models/ and db/.
This is a prototype: lip movement != proven speech, track IDs can switch,
and an ordinary webcam cannot establish physical distance or liveness.

Usage (from FaceRecognition_Capstone project folder):
    .\.venv\Scripts\python.exe combined_demo.py
Quit: press Q in the webcam window.
"""

from collections import deque
from pathlib import Path
import time

import cv2
import mediapipe as mp
import numpy as np
import supervision as sv

# Reuse the functions and registered profiles in your EXISTING face_demo.py.
import face_demo as fd


ROOT = Path(__file__).resolve().parent
FACE_LANDMARKER_MODEL = ROOT / "models" / "face_landmarker.task"

# First-pass demo parameters; calibrate with real recordings.
CONFIRM_FRAMES = 3
MAX_FACES = 4
LIP_HISTORY_FRAMES = 12
LIP_MOTION_THRESHOLD = 0.045
MIN_FACE_ASSOCIATION_IOU = 0.12

# MediaPipe face mesh indices, ordered around the outer lip contour.
LIP_CONTOUR = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
]


def landmark_point(face_landmarks, point_id, width, height):
    p = face_landmarks[point_id]
    return (int(p.x * width), int(p.y * height))


def landmarks_box(face_landmarks, width, height):
    """Approximate face rectangle from all MediaPipe face landmarks."""
    xs = np.array([p.x * width for p in face_landmarks], dtype=np.float32)
    ys = np.array([p.y * height for p in face_landmarks], dtype=np.float32)
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def mouth_opening_ratio(face_landmarks, width, height):
    """Scale-independent-ish vertical mouth gap / horizontal mouth width."""
    upper = np.array(landmark_point(face_landmarks, 13, width, height), dtype=float)
    lower = np.array(landmark_point(face_landmarks, 14, width, height), dtype=float)
    left = np.array(landmark_point(face_landmarks, 61, width, height), dtype=float)
    right = np.array(landmark_point(face_landmarks, 291, width, height), dtype=float)
    return float(np.linalg.norm(lower - upper) / max(np.linalg.norm(right - left), 1.0))


def draw_lips(frame, face_landmarks, moving, width, height):
    points = np.array(
        [landmark_point(face_landmarks, i, width, height) for i in LIP_CONTOUR],
        dtype=np.int32,
    )
    color = (0, 255, 255) if moving else (0, 255, 0)
    cv2.polylines(frame, [points], isClosed=True, color=color, thickness=2)
    for landmark_id in (13, 14, 61, 291):
        cv2.circle(frame, landmark_point(face_landmarks, landmark_id, width, height),
                   3, (0, 0, 255), -1)


def detections_from_faces(faces):
    boxes = np.array([fd.face_box(face) for face in faces], dtype=np.float32).reshape(-1, 4)
    if len(faces) == 0:
        return boxes, sv.Detections.empty()
    return boxes, sv.Detections(
        xyxy=boxes,
        confidence=faces[:, -1].astype(np.float32),
        class_id=np.zeros(len(faces), dtype=int),
    )


def main():
    if not FACE_LANDMARKER_MODEL.is_file():
        raise FileNotFoundError(
            f"Missing MediaPipe model: {FACE_LANDMARKER_MODEL}\n"
            "Download face_landmarker.task into the models folder."
        )

    profiles = fd.load_profiles()
    if not profiles:
        raise RuntimeError("No enrolled faces in db/. First run: python face_demo.py enroll user1")

    if not hasattr(sv, "ByteTrack"):
        raise RuntimeError(
            f"Installed supervision {sv.__version__} has no sv.ByteTrack. "
            "Run: .\\.venv\\Scripts\\python.exe -m pip install supervision==0.25.0"
        )

    tracker = sv.ByteTrack(
        track_activation_threshold=0.25,
        lost_track_buffer=15,
        frame_rate=30,
        minimum_consecutive_frames=1,
    )
    confirmations = {}  # track id -> (latest identified name, consecutive count)
    lip_histories = {}  # track id -> (identified name, deque of mouth opening ratios)
    last_seen = {}      # track id -> frame index; remove old records
    frame_idx = 0
    last_frame_time = time.perf_counter()

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(FACE_LANDMARKER_MODEL)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=MAX_FACES,
    )

    cap = fd.open_camera()
    print("Combined demo running: press Q in camera window to quit.")
    print("Green=registered; Red=unknown/verifying. Lip movement does NOT prove speech.")

    try:
        with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Camera frame could not be read.")
                    break
                frame_idx += 1
                height, width = frame.shape[:2]
                start = time.perf_counter()

                # 1) YuNet: detect every face in this frame.
                faces = fd.find_faces(frame)
                boxes, detections = detections_from_faces(faces)

                # 2) ByteTrack: associate face boxes across frames.
                tracks = tracker.update_with_detections(detections)

                # 3) MediaPipe: locate lips, full-frame (up to MAX_FACES).
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=np.ascontiguousarray(rgb),
                )
                landmarker_result = landmarker.detect(mp_image)
                landmark_faces = landmarker_result.face_landmarks or []
                landmark_boxes = [landmarks_box(lm, width, height) for lm in landmark_faces]
                assigned_landmarks = set()

                view = frame.copy()
                tracked_ids = set()

                if tracks.tracker_id is not None:
                    for box, track_id in zip(tracks.xyxy, tracks.tracker_id):
                        if track_id is None or len(faces) == 0:
                            continue
                        track_id = int(track_id)
                        tracked_ids.add(track_id)
                        last_seen[track_id] = frame_idx

                        # Associate the current track box with its YuNet detection.
                        candidate_scores = [fd.iou(box, raw_box) for raw_box in boxes]
                        detection_idx = int(np.argmax(candidate_scores))
                        if candidate_scores[detection_idx] < 0.35:
                            continue

                        # 4) SFace: verify the identity of THIS tracked face.
                        name, similarity = fd.recognize(frame, faces[detection_idx], profiles)
                        old_name, old_count = confirmations.get(track_id, ("Unknown", 0))
                        count = old_count + 1 if name == old_name else 1
                        confirmations[track_id] = (name, count)
                        confirmed = name != "Unknown" and count >= CONFIRM_FRAMES

                        if not confirmed:
                            lip_histories.pop(track_id, None)
                            fd.draw_box(view, box, f"Track {track_id} | Unknown / verifying", (0, 0, 255))
                            continue

                        # 5) Match the MediaPipe face to THIS recognized track.
                        # Do NOT simply take face_landmarks[0] in multi-person scenes.
                        choice, best_overlap = None, 0.0
                        for j, landmark_box in enumerate(landmark_boxes):
                            if j in assigned_landmarks:
                                continue
                            overlap = fd.iou(box, landmark_box)
                            if overlap > best_overlap:
                                choice, best_overlap = j, overlap

                        if choice is None or best_overlap < MIN_FACE_ASSOCIATION_IOU:
                            lip_histories.pop(track_id, None)
                            fd.draw_box(view, box, f"Track {track_id} | {name} | No lips", (0, 200, 0))
                            continue

                        assigned_landmarks.add(choice)
                        landmarks = landmark_faces[choice]

                        # 6) Compare lip opening over the recent frames.
                        current_ratio = mouth_opening_ratio(landmarks, width, height)
                        history_name, ratios = lip_histories.get(
                            track_id, (name, deque(maxlen=LIP_HISTORY_FRAMES))
                        )
                        if history_name != name:
                            ratios = deque(maxlen=LIP_HISTORY_FRAMES)
                        ratios.append(current_ratio)
                        lip_histories[track_id] = (name, ratios)
                        moving = (
                            len(ratios) >= 8
                            and max(ratios) - min(ratios) > LIP_MOTION_THRESHOLD
                        )

                        draw_lips(view, landmarks, moving, width, height)
                        mouth_state = "LIP MOVING" if moving else "LIP STILL"
                        fd.draw_box(view, box,
                                    f"Track {track_id} | {name} ({similarity:.2f}) | {mouth_state}",
                                    (0, 200, 0))
                        bx, by = int(box[0]), int(box[3])
                        cv2.putText(view, f"Mouth ratio: {current_ratio:.3f}",
                                    (max(0, bx), min(height - 10, by + 25)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)

                # Discard stale state when tracks disappear. This is not identity security.
                for tid in list(last_seen):
                    if frame_idx - last_seen[tid] > 45:
                        last_seen.pop(tid, None)
                        confirmations.pop(tid, None)
                        lip_histories.pop(tid, None)

                now = time.perf_counter()
                fps = 1.0 / max(now - last_frame_time, 1e-6)
                last_frame_time = now
                processing_ms = (now - start) * 1000
                cv2.putText(view, f"FPS {fps:.1f} | Processing {processing_ms:.0f} ms",
                            (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                cv2.putText(view, "Registered face -> lip motion | Not speech/distance | Q quit",
                            (15, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 255, 255), 2)
                cv2.imshow("YuNet + SFace + ByteTrack + MediaPipe Lips", view)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()