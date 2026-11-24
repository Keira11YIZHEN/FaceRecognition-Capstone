# FaceRecognition-Capstone

**NTU Capstone prototype:** live face detection and recognition, tracking, lip motion tracking, and microphone voice activity detection.

## Components

| Script | Function |
|---|---|
| `face_demo.py` | YuNet face detection, SFace identity recognition and enrollment, ByteTrack tracking |
| `combined_demo.py` | Integrates recognition/tracking with MediaPipe mouth landmarks |
| `demo_showcase.py` | Status dashboard for a group presentation |
| `demo_av_fusion.py` | Webcam + Silero VAD microphone fusion demo |

Keep **all four Python files** in the same folder: the later scripts import the earlier ones.

## Download model files

Create a `models` directory and download these three files into it:

- `face_detection_yunet_2023mar.onnx`: https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
- `face_recognition_sface_2021dec.onnx`: https://github.com/opencv/opencv_zoo/blob/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
- `face_landmarker.task`: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

For GitHub model files, select **Download raw file**, not the HTML page. Model binaries are intentionally excluded from the repository.

## Windows setup

The original project used **Python 3.13**. `requirements.txt` is a **starter dependency list inferred from source code**, not the original computer's exact `pip freeze` export.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Check ByteTrack compatibility if needed:

```powershell
.\.venv\Scripts\python.exe -c "import supervision as sv; print(sv.__version__, hasattr(sv, 'ByteTrack'))"
```

## Register a test user

```powershell
.\.venv\Scripts\python.exe face_demo.py enroll user1
```

A webcam window opens. Press **C** five times to capture enrollment samples. The file `db/user1.npz` is saved only on the local computer and **must not be uploaded**.

## Launch final demo

```powershell
.\.venv\Scripts\python.exe -u demo_av_fusion.py
```

Optional commands:

```powershell
.\.venv\Scripts\python.exe -u demo_av_fusion.py --list-mics
.\.venv\Scripts\python.exe -u demo_av_fusion.py --mic 2
.\.venv\Scripts\python.exe demo_showcase.py
.\.venv\Scripts\python.exe face_demo.py run
```

In the video window, **Q** exits. **S** saves a screenshot in the showcase modes.

## Limitations and privacy

- **Distance: Not Measured.** A normal webcam does not supply reliable physical distance. D435i depth integration is future work.
- **Speech: Not Verified.** Silero VAD detects speech in nearby audio; lip motion plus speech only implies **possible** target speech. It does **not** identify the speaker and is not yet an audiovisual active-speaker verification model.
- SFace matching is a research prototype rather than a secure access-control solution.
- Never upload `.venv/`, `db/`, face embeddings, secrets, voice recordings, or camera screenshots. These paths are excluded in `.gitignore`.
- The source is uploaded for group collaboration; it has not been tested on every collaborator's machine.
