"""MediaPipe: modelos oficiales, silueta de la persona y detector del gesto."""

from __future__ import annotations

import os
import urllib.request
from collections.abc import Callable
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from .gesture import two_fingers_on_forehead
from .paths import MODEL_DIR

_BASE = "https://storage.googleapis.com/mediapipe-models"
MODEL_URLS = {
    "selfie_segmenter_landscape.tflite":
        f"{_BASE}/image_segmenter/selfie_segmenter_landscape/float16/latest/"
        "selfie_segmenter_landscape.tflite",
    "hand_landmarker.task":
        f"{_BASE}/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    "face_landmarker.task":
        f"{_BASE}/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
}


def ensure_models(model_dir: Path = MODEL_DIR,
                  progress: Callable[[str, float], None] | None = None) -> dict[str, Path]:
    """Descarga lo que falte (solo la primera vez) y devuelve la ruta de cada modelo."""
    model_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, url in MODEL_URLS.items():
        path = model_dir / name
        paths[name] = path
        if path.is_file() and path.stat().st_size > 100_000:
            continue
        partial = path.with_suffix(path.suffix + ".part")
        request = urllib.request.Request(url, headers={"User-Agent": "helcam"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as out:
                total = int(response.headers.get("Content-Length") or 0)
                received = 0
                while chunk := response.read(1 << 18):
                    out.write(chunk)
                    received += len(chunk)
                    if progress and total:
                        progress(name, received / total)
            os.replace(partial, path)
        except OSError as error:
            partial.unlink(missing_ok=True)
            raise RuntimeError(f"No pude descargar {name}. Revisa tu conexión a internet.") from error
    return paths


def _rgb_image(frame_bgr: np.ndarray) -> mp.Image:
    return mp.Image(image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))


class _Clock:
    """MediaPipe en modo VIDEO exige marcas de tiempo estrictamente crecientes."""

    def __init__(self):
        self.last = -1

    def next(self, timestamp_ms: int) -> int:
        self.last = max(timestamp_ms, self.last + 1)
        return self.last


class PersonSegmenter:
    def __init__(self, model: Path):
        options = mp_vision.ImageSegmenterOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model)),
            running_mode=mp_vision.RunningMode.VIDEO,
            output_confidence_masks=True,
            output_category_mask=False,
        )
        self._task = mp_vision.ImageSegmenter.create_from_options(options)
        self._clock = _Clock()

    def mask(self, frame_bgr: np.ndarray, timestamp_ms: int) -> np.ndarray:
        """Máscara 0..1 de la persona, del tamaño del cuadro y con bordes suaves."""
        height, width = frame_bgr.shape[:2]
        result = self._task.segment_for_video(_rgb_image(frame_bgr), self._clock.next(timestamp_ms))
        if not result.confidence_masks:
            return np.zeros((height, width), np.float32)
        confidence = np.squeeze(result.confidence_masks[0].numpy_view()).astype(np.float32)
        confidence = cv2.resize(confidence, (width, height), interpolation=cv2.INTER_LINEAR)
        confidence = cv2.GaussianBlur(confidence, (0, 0), sigmaX=2.0)
        # Rampa entre 0.2 y 0.75: borra el halo dudoso y conserva el pelo.
        return np.clip((confidence - 0.2) / 0.55, 0.0, 1.0)

    def close(self) -> None:
        self._task.close()


class GokuPoseDetector:
    """Manos y cara del mismo cuadro: ¿hay dos dedos juntos sobre la frente?"""

    def __init__(self, hand_model: Path, face_model: Path):
        self._hands = mp_vision.HandLandmarker.create_from_options(mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(hand_model)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.35,
            min_hand_presence_confidence=0.35,
            min_tracking_confidence=0.45,
        ))
        try:
            self._face = mp_vision.FaceLandmarker.create_from_options(mp_vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(face_model)),
                running_mode=mp_vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.4,
                min_face_presence_confidence=0.4,
                min_tracking_confidence=0.4,
            ))
        except Exception:
            self._hands.close()
            raise
        self._clock = _Clock()
        self.hands_seen = 0
        self.face_seen = False

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> bool:
        image = _rgb_image(frame_bgr)
        timestamp_ms = self._clock.next(timestamp_ms)
        hands = self._hands.detect_for_video(image, timestamp_ms)
        faces = self._face.detect_for_video(image, timestamp_ms)
        self.hands_seen = len(hands.hand_landmarks)
        self.face_seen = bool(faces.face_landmarks)
        if not self.face_seen:
            return False
        return two_fingers_on_forehead(hands.hand_landmarks, hands.hand_world_landmarks,
                                       faces.face_landmarks[0], image.width, image.height)

    def close(self) -> None:
        try:
            self._hands.close()
        finally:
            self._face.close()
