"""Gesto de Goku: indice y medio juntos sobre la frente, sostenidos un segundo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


HOLD_SECONDS = 1.0
# Cortes breves del tracking no reinician la cuenta.
RELEASE_SECONDS = 0.5

# Índices del face landmarker de MediaPipe (468 puntos).
FOREHEAD_TOP = 10
BROW_CENTER = 9
FACE_LEFT = 234
FACE_RIGHT = 454


def _finger(points: np.ndarray, base: int) -> tuple[float, float]:
    """Relación cuerda/largo (1 = dedo recto) y largo del dedo."""
    finger = points[base : base + 4]
    length = float(np.linalg.norm(np.diff(finger, axis=0), axis=1).sum())
    if length < 1e-6:
        return 0.0, 0.0
    return float(np.linalg.norm(finger[3] - finger[0])) / length, length


def two_finger_pose(points: np.ndarray) -> bool:
    """Índice y medio estirados y juntos, anular y meñique doblados."""
    if points.shape[0] != 21 or not np.isfinite(points).all():
        return False
    index_ratio, index_length = _finger(points, 5)
    middle_ratio, middle_length = _finger(points, 9)
    reference = min(index_length, middle_length)
    if reference < 1e-6 or min(index_ratio, middle_ratio) < 0.80:
        return False
    for base in (13, 17):
        ratio, _ = _finger(points, base)
        reach = float(np.linalg.norm(points[base + 3] - points[base]))
        if ratio >= 0.85 and reach >= reference * 0.75:
            return False
    # Juntos, no en V: las puntas quedan cerca.
    return float(np.linalg.norm(points[8] - points[12])) < reference * 0.38


def _segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    direction = end - start
    span = float(np.dot(direction, direction))
    if span < 1e-9:
        return float(np.linalg.norm(point - start))
    t = float(np.clip(np.dot(point - start, direction) / span, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + direction * t)))


def two_fingers_on_forehead(hands, world_hands, face, width: int, height: int) -> bool:
    """Las puntas de índice y medio tocan la frente, entre las cejas y el nacimiento del pelo."""
    if not face or len(face) <= FACE_RIGHT:
        return False
    face_points = np.array([(p.x * width, p.y * height) for p in face])
    face_width = float(np.linalg.norm(face_points[FACE_LEFT] - face_points[FACE_RIGHT]))
    if face_width < 20.0:
        return False
    top, bottom = face_points[FOREHEAD_TOP], face_points[BROW_CENTER]
    for number, hand in enumerate(hands):
        if len(hand) != 21:
            continue
        image_points = np.array([(p.x * width, p.y * height) for p in hand])
        # La geometría 3D no se aplana cuando los dedos apuntan hacia la cámara.
        if number < len(world_hands) and len(world_hands[number]) == 21:
            shape = np.array([(p.x, p.y, p.z) for p in world_hands[number]])
        else:
            shape = np.column_stack((image_points, np.zeros(21)))
        if not two_finger_pose(shape):
            continue
        # Las yemas también cuentan: con los dedos apoyados de plano cubren la frente.
        reach = min(_segment_distance(image_points[i], top, bottom) for i in (7, 8, 11, 12))
        if reach <= face_width * 0.22:
            return True
    return False


@dataclass
class HoldTimer:
    """Mide cuánto dura el gesto; después de disparar hay que bajar la mano."""

    hold_seconds: float = HOLD_SECONDS
    release_seconds: float = RELEASE_SECONDS
    started: float | None = None
    last_seen: float = float("-inf")
    armed: bool = True

    def update(self, detected: bool, now: float) -> str | None:
        """Devuelve "started", "lost", "triggered" o None."""
        if detected:
            self.last_seen = now
            if not self.armed:
                return None
            if self.started is None:
                self.started = now
                return "started"
            if now - self.started >= self.hold_seconds:
                self.started = None
                self.armed = False
                return "triggered"
            return None
        if now - self.last_seen <= self.release_seconds:
            return None
        self.armed = True
        if self.started is not None:
            self.started = None
            return "lost"
        return None

    def remaining(self, now: float) -> float:
        if self.started is None:
            return self.hold_seconds
        return max(0.0, self.hold_seconds - (now - self.started))

    def reset(self) -> None:
        self.started = None
        self.last_seen = float("-inf")
        self.armed = True

