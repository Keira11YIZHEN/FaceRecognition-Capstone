# -*- coding: utf-8 -*-
r"""NTU Capstone audio-visual progress demo (webcam + microphone).

Requirements (in this project folder):
    face_demo.py, combined_demo.py
    models/face_detection_yunet_2023mar.onnx
    models/face_recognition_sface_2021dec.onnx
    models/face_landmarker.task
    db/user1.npz  (or another registered .npz profile)

Install audio dependencies into the project's existing .venv:
    .\.venv\Scripts\python.exe -m pip install sounddevice silero-vad

Run:
    .\.venv\Scripts\python.exe -u demo_av_fusion.py
    .\.venv\Scripts\python.exe -u demo_av_fusion.py --list-mics
    .\.venv\Scripts\python.exe -u demo_av_fusion.py --mic 2

Q or Esc = quit; S = screenshot; no recording is saved by default.

CAUTION: VAD detects human speech near the microphone, NOT the speaker's
identity. Lip motion + nearby speech is only a heuristic for POSSIBLE
speaking and must never be labelled verified. Distance is not measured.
"""

print('[1/4] Loading face recognition, tracking and lip modules...', flush=True)

import argparse
from collections import deque
from datetime import datetime
from pathlib import Path
import queue
import threading
import time

import cv2
import numpy as np
import mediapipe as mp
import supervision as sv

import demo_showcase as vis   # Reuse the existing visual layout and helpers

core = vis.core
fd = vis.fd
ROOT = Path(__file__).resolve().parent
MODEL = ROOT / 'models' / 'face_landmarker.task'
OUT_DIR = ROOT / 'captures'

# These are first-pass DEMO thresholds, not validated speaker identity rules.
CONFIRM_FRAMES = 3
LIP_HISTORY = 12
LIP_THRESHOLD = 0.045
MAX_FACES = 4
MIN_FACE_IOU = 0.35
MIN_LIP_IOU = 0.12

VIDEO_W = 900
VIDEO_H = 650
PANEL_W = 385

DARK = (34, 28, 22)
CARD = (48, 42, 37)
WHITE = (242, 242, 242)
MUTED = (177, 173, 165)
GREEN = (117, 220, 108)
YELLOW = (112, 211, 253)
RED = (112, 111, 242)
BLUE = (240, 181, 110)


def draw_text(canvas, text, x, y, scale=0.52, color=WHITE, thickness=1):
    cv2.putText(canvas, str(text), (int(x), int(y)),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


class AudioSpeechMonitor:
    """Listen and run Silero VAD off the camera/UI thread.

    Uses sounddevice.InputStream callback -> bounded queue -> worker VAD.
    16 kHz mono, 0.25 s blocks; analyze approximately 1 s rolling audio.
    Failures show MIC ERROR; video keeps running.
    """

    def __init__(self, mic_index=None, enabled=True):
        self.mic_index = mic_index
        self.enabled = enabled
        self.stop_event = threading.Event()
        self._lock = threading.Lock()
        self._status = 'LOADING' if enabled else 'DISABLED'
        self._error = ''
        self._last_speech_at = float('-inf')
        self._last_processed_at = float('-inf')
        self._thread = None

    def start(self):
        if not self.enabled:
            return
        self._thread = threading.Thread(target=self._worker, name='speech-vad', daemon=True)
        self._thread.start()

    def close(self):
        self.stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.5)

    def snapshot(self):
        with self._lock:
            status, error = self._status, self._error
            last = self._last_speech_at
            last_processed = self._last_processed_at
        # The hold prevents rapid flicker while processing fresh audio.
        active = (status == 'READY' and (time.monotonic() - last) < 0.9)
        stale = status == 'READY' and (time.monotonic() - last_processed) > 3.0
        if stale:
            return 'MIC STALLED', False, error
        if status == 'READY':
            return ('DETECTED' if active else 'NO SPEECH'), active, ''
        return status, False, error

    def _set_status(self, status, error=''):
        with self._lock:
            self._status = status
            self._error = error
            if status == 'READY':
                self._last_processed_at = time.monotonic()

    def _worker(self):
        try:
            print('[AUDIO] Loading Silero VAD...', flush=True)
            import sounddevice as sd
            import torch
            from silero_vad import load_silero_vad, get_speech_timestamps

            # Avoid many CPU threads competing with the live camera inference.
            torch.set_num_threads(1)
            model = load_silero_vad()
            audio_queue = queue.Queue(maxsize=12)

            def callback(indata, frames, callback_time, status):
                # Keep callback minimal; all ML inference happens in worker thread.
                if self.stop_event.is_set():
                    return
                packet = np.ascontiguousarray(indata[:, 0].copy(), dtype=np.float32)
                try:
                    audio_queue.put_nowait(packet)
                except queue.Full:
                    try:
                        audio_queue.get_nowait()  # Drop oldest delayed packet.
                    except queue.Empty:
                        pass
                    try:
                        audio_queue.put_nowait(packet)
                    except queue.Full:
                        pass

            print('[AUDIO] Opening microphone at 16000 Hz...', flush=True)
            with sd.InputStream(samplerate=16000, channels=1, dtype='float32',
                                blocksize=4096, device=self.mic_index,
                                callback=callback):
                self._set_status('READY')
                print('[AUDIO] Microphone ready; listening (no audio saved).', flush=True)
                window = deque(maxlen=4)  # approx 1.02 seconds
                packets_since_infer = 0
                while not self.stop_event.is_set():
                    try:
                        packet = audio_queue.get(timeout=0.4)
                    except queue.Empty:
                        continue
                    window.append(packet)
                    packets_since_infer += 1
                    if len(window) < 2 or packets_since_infer < 2:
                        continue
                    packets_since_infer = 0
                    audio = np.ascontiguousarray(np.concatenate(tuple(window)), dtype=np.float32)
                    # Current window overlaps previous one; this is a proximity
                    # heuristic, not synchronized biometric speaker verification.
                    with torch.inference_mode():
                        segments = get_speech_timestamps(
                            torch.from_numpy(audio), model,
                            sampling_rate=16000,
                            min_speech_duration_ms=150,
                            min_silence_duration_ms=100,
                        )
                    now = time.monotonic()
                    recent = any(segment['end'] >= len(audio) - 8000 for segment in segments)
                    with self._lock:
                        self._last_processed_at = now
                        if recent:
                            self._last_speech_at = now
        except Exception as exc:
            message = f'{type(exc).__name__}: {exc}'
            self._set_status('MIC ERROR', message)
            print(f'[AUDIO] {message}', flush=True)
            print('[AUDIO] Video continues. Try --list-mics or Windows microphone permissions.', flush=True)


def draw_panel(state):
    panel = np.full((VIDEO_H, PANEL_W, 3), DARK, dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (PANEL_W, 76), CARD, -1)
    draw_text(panel, 'CAPSTONE  /  AUDIO-VISUAL DEMO', 18, 29, 0.57, WHITE, 2)
    draw_text(panel, 'YuNet | SFace | ByteTrack | MediaPipe | VAD', 18, 56, 0.38, MUTED)

    items = [
        ('FACE DETECTED', 'YES' if state['faces'] else 'NO',
         GREEN if state['faces'] else MUTED),
        ('IDENTITY', state['identity'],
         GREEN if state['identity'] not in ('NONE', 'UNKNOWN', 'VERIFYING')
         else YELLOW if state['identity'] == 'VERIFYING' else MUTED),
        ('TRACK ID', state['track'], BLUE if state['track'] != '--' else MUTED),
        ('LIP MOTION', state['lip'],
         YELLOW if state['lip'] == 'MOVING' else GREEN if state['lip'] == 'STILL' else MUTED),
        ('MICROPHONE SPEECH', state['audio'],
         GREEN if state['audio'] == 'DETECTED' else RED if state['audio'] in ('MIC ERROR', 'MIC STALLED') else MUTED),
        ('TARGET SPEAKING', state['speaking'],
         YELLOW if state['speaking'] == 'POSSIBLE' else MUTED),
        ('FPS / PROCESS', f"{state['fps']:.1f} FPS   /   {state['latency']:.0f} ms", WHITE),
        ('FACES IN FRAME', str(state['faces']), WHITE),
    ]
    y0 = 98
    row_h = 59
    for i, (label, value, color) in enumerate(items):
        y = y0 + i * row_h
        draw_text(panel, label, 22, y, 0.40, MUTED)
        draw_text(panel, value, 22, y + 26, 0.65, color, 2)
        cv2.line(panel, (20, y + 38), (PANEL_W - 20, y + 38), CARD, 1)

    cv2.rectangle(panel, (12, VIDEO_H - 75), (PANEL_W - 12, VIDEO_H - 13), CARD, -1)
    draw_text(panel, 'DISTANCE: NOT MEASURED', 22, VIDEO_H - 51, 0.46, YELLOW)
    draw_text(panel, 'SPEAKER ID: NOT VERIFIED', 22, VIDEO_H - 27, 0.46, YELLOW)
    return panel


def decide_target_speech(identity, lip_status, audio_status, audio_detected):
    """Transparent demo heuristic; never return 'verified'."""
    if identity in ('NONE', 'UNKNOWN', 'VERIFYING'):
        return 'NOT VERIFIED'
    if audio_status not in ('DETECTED', 'NO SPEECH'):
        return 'WAITING FOR MIC'
    if lip_status == 'MOVING' and audio_detected:
        return 'POSSIBLE'
    if lip_status == 'MOVING' and not audio_detected:
        return 'MOUTH ONLY'
    if audio_detected:
        return 'SPEAKER UNCLEAR'
    if lip_status in ('STILL', 'CALIBRATING'):
        return 'NO SPEECH EVIDENCE' if lip_status == 'STILL' else 'WAITING FOR LIPS'
    return 'NOT VERIFIED'


def letterbox_video(frame):
    """Keep camera aspect ratio instead of stretching faces in the demo."""
    h, w = frame.shape[:2]
    scale = min(VIDEO_W / w, (VIDEO_H - 66) / h)
    dw, dh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    canvas = np.full((VIDEO_H, VIDEO_W, 3), DARK, dtype=np.uint8)
    top = 43 + max(0, ((VIDEO_H - 66) - dh) // 2)
    left = (VIDEO_W - dw) // 2
    canvas[top:top + dh, left:left + dw] = resized
    cv2.rectangle(canvas, (0, 0), (VIDEO_W, 40), DARK, -1)
    draw_text(canvas, 'LIVE CAMERA  |  FACE + LIPS + MICROPHONE', 15, 26, 0.56, WHITE, 2)
    cv2.rectangle(canvas, (0, VIDEO_H - 26), (VIDEO_W, VIDEO_H), DARK, -1)
    draw_text(canvas, 'Q / Esc: exit     S: screenshot     Audio+lip = possible, NOT verified',
              15, VIDEO_H - 8, 0.45, YELLOW)
    return canvas


def run(mic_index=None, no_audio=False):
    for model_path in (fd.YUNET, fd.SFACE, MODEL):
        if not model_path.is_file():
            raise FileNotFoundError(f'Missing model: {model_path}')
    if not hasattr(sv, 'ByteTrack'):
        raise RuntimeError('sv.ByteTrack unavailable. Install supervision==0.25.0')

    profiles = fd.load_profiles()
    if not profiles:
        raise RuntimeError('No registered profile in db/. Run face_demo.py enroll user1 first.')

    tracker = sv.ByteTrack(track_activation_threshold=0.25,
                           lost_track_buffer=15, frame_rate=30,
                           minimum_consecutive_frames=1)
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=MAX_FACES,
    )
    audio = AudioSpeechMonitor(mic_index=mic_index, enabled=not no_audio)
    cap = None
    audio.start()  # startup on background thread; video remains responsive
    print('[2/4] Loading camera...', flush=True)

    confirmations = {}
    lip_histories = {}
    last_seen = {}
    frame_idx = 0
    last_fps_time = time.perf_counter()
    fps_ema = 0.0
    try:
        cap = fd.open_camera()
        print('[3/4] Starting MediaPipe Face Landmarker...', flush=True)
        with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
            print('[4/4] Demo ready! Q/Esc=quit, S=save screenshot.', flush=True)
            while True:
                ok, frame = cap.read()
                if not ok:
                    print('Could not read webcam frame.', flush=True)
                    break
                frame_idx += 1
                h, w = frame.shape[:2]
                start = time.perf_counter()

                faces = fd.find_faces(frame)
                boxes, detections = core.detections_from_faces(faces)
                tracks = tracker.update_with_detections(detections)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                    data=np.ascontiguousarray(rgb))
                result = landmarker.detect(mp_image)
                landmark_faces = result.face_landmarks or []
                landmark_boxes = [core.landmarks_box(lm, w, h) for lm in landmark_faces]
                assigned_landmarks = set()

                view = frame.copy()
                featured = None
                if tracks.tracker_id is not None:
                    for box, raw_id in zip(tracks.xyxy, tracks.tracker_id):
                        if raw_id is None or not len(faces):
                            continue
                        tid = int(raw_id)
                        last_seen[tid] = frame_idx
                        scores = [fd.iou(box, current) for current in boxes]
                        idx = int(np.argmax(scores))
                        if scores[idx] < MIN_FACE_IOU:
                            continue

                        name, similarity = fd.recognize(frame, faces[idx], profiles)
                        last_name, last_count = confirmations.get(tid, ('Unknown', 0))
                        count = last_count + 1 if last_name == name else 1
                        confirmations[tid] = (name, count)
                        known = name != 'Unknown' and count >= CONFIRM_FRAMES
                        identity = name if known else ('UNKNOWN' if name == 'Unknown' else 'VERIFYING')
                        lip = 'NOT ANALYZED'

                        if known:
                            nearest = None
                            highest_iou = 0.0
                            for face_i, land_box in enumerate(landmark_boxes):
                                if face_i in assigned_landmarks:
                                    continue
                                score = fd.iou(box, land_box)
                                if score > highest_iou:
                                    nearest, highest_iou = face_i, score
                            if nearest is not None and highest_iou >= MIN_LIP_IOU:
                                assigned_landmarks.add(nearest)
                                lm = landmark_faces[nearest]
                                opening = core.mouth_opening_ratio(lm, w, h)
                                old_lip_name, ratios = lip_histories.get(tid, (name, deque(maxlen=LIP_HISTORY)))
                                if old_lip_name != name:
                                    ratios = deque(maxlen=LIP_HISTORY)
                                ratios.append(opening)
                                lip_histories[tid] = (name, ratios)
                                moving = (len(ratios) >= 8 and
                                          max(ratios) - min(ratios) > LIP_THRESHOLD)
                                lip = ('MOVING' if moving else 'STILL' if len(ratios) >= 8
                                       else 'CALIBRATING')
                                core.draw_lips(view, lm, moving, w, h)
                            else:
                                lip_histories.pop(tid, None)
                                lip = 'NO LANDMARKS'
                        else:
                            lip_histories.pop(tid, None)

                        vis.paint_face(view, box, identity, tid, lip, similarity)
                        area = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
                        priority = 2 if known else 1
                        candidate = {'identity': identity, 'track': str(tid), 'lip': lip}
                        if featured is None or (priority, area) > featured[:2]:
                            featured = (priority, area, candidate)

                for tid in list(last_seen):
                    if frame_idx - last_seen[tid] > 45:
                        last_seen.pop(tid, None)
                        confirmations.pop(tid, None)
                        lip_histories.pop(tid, None)

                audio_status, audio_detected, audio_error = audio.snapshot()
                state = {
                    'faces': len(faces), 'identity': 'NONE', 'track': '--',
                    'lip': 'WAITING', 'audio': audio_status,
                    'fps': fps_ema, 'latency': 0.0,
                }
                if featured:
                    state.update(featured[2])
                state['speaking'] = decide_target_speech(
                    state['identity'], state['lip'], audio_status, audio_detected)

                end = time.perf_counter()
                fps = 1.0 / max(end - last_fps_time, 1e-6)
                fps_ema = 0.8 * fps_ema + 0.2 * fps if fps_ema else fps
                last_fps_time = end
                state['fps'] = fps_ema
                state['latency'] = (end - start) * 1000

                canvas = letterbox_video(view)
                panel = draw_panel(state)
                screen = np.hstack((canvas, panel))
                cv2.imshow('NTU Capstone  |  Audio-Visual Progress Demo', screen)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
                if key == ord('s'):
                    OUT_DIR.mkdir(exist_ok=True)
                    name = f"av_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    output = OUT_DIR / name
                    if cv2.imwrite(str(output), screen):
                        print(f'[SCREENSHOT] Saved: {output}', flush=True)
                    else:
                        print(f'[SCREENSHOT] Failed: {output}', flush=True)
    finally:
        audio.close()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='NTU Capstone: Face + Lip + Audio VAD Demo')
    parser.add_argument('--mic', type=int, default=None,
                        help='Sounddevice microphone device index; defaults to system input.')
    parser.add_argument('--list-mics', action='store_true',
                        help='List microphones and their indices, then quit.')
    parser.add_argument('--no-audio', action='store_true',
                        help='Show vision only; useful for microphone troubleshooting.')
    args = parser.parse_args()
    if args.list_mics:
        import sounddevice as sd
        for i, dev in enumerate(sd.query_devices()):
            if dev['max_input_channels'] > 0:
                print(f"{i}: {dev['name']}  |  input channels={dev['max_input_channels']}")
        return
    run(mic_index=args.mic, no_audio=args.no_audio)


if __name__ == '__main__':
    main()