"""Capstone prototype: YuNet + SFace + ByteTrack.

This is a local webcam experiment, NOT a production authentication service.
It cannot verify the physical 1.5 m distance and only PRINTS a greeting event.
"""

import sys
import re
import time
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
DB = ROOT / "db"
DB.mkdir(exist_ok=True)

YUNET = MODELS / "face_detection_yunet_2023mar.onnx"
SFACE = MODELS / "face_recognition_sface_2021dec.onnx"

# Demonstration threshold from the OpenCV SFace example.
# Calibrate with separate validation subjects before using for access control.
MATCH_THRESHOLD = 0.363
DETECT_WIDTH = 640
CONFIRM_FRAMES = 3

for model_path in (YUNET, SFACE):
    if not model_path.is_file():
        raise FileNotFoundError(f"Model missing: {model_path}")

detector = cv2.FaceDetectorYN.create(
    str(YUNET), "", (320, 320)
)
detector.setScoreThreshold(0.70)
recognizer = cv2.FaceRecognizerSF.create(str(SFACE), "")


def open_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera 0 cannot be opened")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap


def find_faces(frame):
    height, width = frame.shape[:2]
    scale = min(1.0, DETECT_WIDTH / width)
    small = cv2.resize(frame, (round(width * scale), round(height * scale))) if scale < 1.0 else frame
    detector.setInputSize((small.shape[1], small.shape[0]))
    _, faces = detector.detect(small)
    if faces is None:
        return np.empty((0, 15), dtype=np.float32)
    faces = faces.astype(np.float32).copy()
    faces[:, :14] /= scale  # Bounding box and five 2D landmarks -> original image coordinates.
    return faces


def embedding(frame, face):
    aligned = recognizer.alignCrop(frame, face)
    return np.ascontiguousarray(recognizer.feature(aligned), dtype=np.float32)


def face_box(face):
    x, y, w, h = face[:4]
    return np.array([x, y, x + w, y + h], dtype=np.float32)


def iou(a, b):
    x1, y1 = np.maximum(a[:2], b[:2])
    x2, y2 = np.minimum(a[2:], b[2:])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return float(inter / max(area_a + area_b - inter, 1e-6))


def draw_box(frame, box, label, color):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (max(0, x1), max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def enroll(name):
    if not re.fullmatch(r"[A-Za-z0-9_]{1,32}", name):
        raise ValueError("Name must contain only English letters, digits or underscore")
    cap = open_camera()
    samples = []
    print(f"Registering: {name}. C=capture (5 samples), Q=quit")
    try:
        while len(samples) < 5:
            ok, frame = cap.read()
            if not ok:
                break
            faces = find_faces(frame)
            view = frame.copy()
            for face in faces:
                draw_box(view, face_box(face), f"Face score {face[-1]:.2f}", (0, 255, 0))
            cv2.putText(view, f"{name}: {len(samples)}/5 | C=capture Q=quit", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            cv2.imshow("Enrollment", view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                if len(faces) != 1:
                    print("Keep exactly one visible face when capturing.")
                    continue
                samples.append(embedding(frame, faces[0]).reshape(-1).copy())
                print(f"Captured {len(samples)}/5")
        if len(samples) == 5:
            output = DB / f"{name}.npz"
            np.savez_compressed(output, embeddings=np.stack(samples))
            print(f"Enrollment saved: {output}")
        else:
            print("Enrollment cancelled/incomplete; nothing saved.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


def load_profiles():
    profiles = {}
    for path in DB.glob("*.npz"):
        with np.load(path, allow_pickle=False) as data:
            profiles[path.stem] = data["embeddings"].astype(np.float32)
    return profiles


def recognize(frame, face, profiles):
    feat = embedding(frame, face)
    best_name, best_score = "Unknown", -1.0
    for name, stored_features in profiles.items():
        for stored in stored_features:
            ref = np.ascontiguousarray(stored.reshape(1, -1), dtype=np.float32)
            score = float(recognizer.match(feat, ref, cv2.FaceRecognizerSF_FR_COSINE))
            if score > best_score:
                best_name, best_score = name, score
    if best_score < MATCH_THRESHOLD:
        return "Unknown", best_score
    return best_name, best_score


def run():
    profiles = load_profiles()
    if not profiles:
        raise RuntimeError("No registered profiles. Run 'enroll user1' first.")
    tracker = sv.ByteTrack(track_activation_threshold=0.25,
                           lost_track_buffer=15, frame_rate=30,
                           minimum_consecutive_frames=1)
    confirmations, first_seen, greeted_tracks = {}, {}, set()
    cap = open_camera()
    last_frame = time.perf_counter()
    print("Live test: Q=quit. Distance is NOT measured by a normal webcam.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t0 = time.perf_counter()
            faces = find_faces(frame)
            boxes = np.array([face_box(face) for face in faces],
                             dtype=np.float32).reshape(-1, 4)
            if len(faces):
                detections = sv.Detections(
                    xyxy=boxes,
                    confidence=faces[:, -1].astype(np.float32),
                    class_id=np.zeros(len(faces), dtype=int)
                )
            else:
                detections = sv.Detections.empty()
            tracks = tracker.update_with_detections(detections)
            view = frame.copy()

            if tracks.tracker_id is not None:
                for box, track_id in zip(tracks.xyxy, tracks.tracker_id):
                    if track_id is None or not len(faces):
                        continue
                    tid = int(track_id)
                    scores = [iou(box, raw_box) for raw_box in boxes]
                    idx = int(np.argmax(scores))
                    if scores[idx] < 0.35:
                        continue
                    first_seen.setdefault(tid, t0)
                    name, score = recognize(frame, faces[idx], profiles)
                    old_name, old_count = confirmations.get(tid, ("Unknown", 0))
                    count = old_count + 1 if name == old_name else 1
                    confirmations[tid] = (name, count)
                    authorized = name != "Unknown" and count >= CONFIRM_FRAMES
                    label = (f"Track {tid} | {name} ({score:.2f})" if authorized
                             else f"Track {tid} | Unknown / verifying")
                    draw_box(view, box, label, (0, 200, 0) if authorized else (0, 0, 255))
                    if authorized and tid not in greeted_tracks:
                        greeted_tracks.add(tid)
                        # Not end-to-end 1.5m-to-spoken-audio latency.
                        delay = time.perf_counter() - first_seen[tid]
                        print(f"GREETING {name} | simulated recognition delay: {delay:.3f} s")

            processing_ms = (time.perf_counter() - t0) * 1000
            now = time.perf_counter()
            fps = 1.0 / max(now - last_frame, 1e-6)
            last_frame = now
            cv2.putText(view, f"FPS {fps:.1f} | Processing {processing_ms:.0f} ms",
                        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(view, "Distance NOT verified | Q=quit", (15, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("YuNet + SFace + ByteTrack", view)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "enroll":
        enroll(sys.argv[2])
    elif len(sys.argv) == 2 and sys.argv[1] == "run":
        run()
    else:
        print("Usage: python face_demo.py enroll NAME | run")