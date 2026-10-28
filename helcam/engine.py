"""Máquina de estados: fondo automático, gesto, desaparecer y reaparecer.

No sabe nada de Qt ni de cámaras: recibe cuadros BGR y devuelve el cuadro limpio
que se manda a la cámara virtual, junto con el estado para la interfaz.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import numpy as np

from .effects import (
    ANIME_SECONDS, GOKU_SECONDS, SNAP_END, make_aura_noise, make_streaks, render_aura,
    render_instant, render_teleport,
)
from .gesture import HoldTimer


class Phase(str, Enum):
    WAITING = "waiting"        # Esperando que la persona salga del cuadro.
    CAPTURING = "capturing"    # Escena vacía: juntando cuadros para el fondo.
    READY = "ready"
    VANISHING = "vanishing"
    HIDDEN = "hidden"
    RETURNING = "returning"


class Style(str, Enum):
    ANIME = "anime"
    VFX = "vfx"


class Segmenter(Protocol):
    def mask(self, frame_bgr: np.ndarray, timestamp_ms: int) -> np.ndarray: ...


class PoseDetector(Protocol):
    hands_seen: int
    face_seen: bool

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> bool: ...


@dataclass(frozen=True)
class Status:
    phase: Phase
    hold: float = 0.0              # 0..1 mientras se sostiene el gesto.
    hands: int = 0
    face: bool = False
    empty_for: float = 0.0         # Segundos con la escena vacía (WAITING).
    capture: float = 0.0           # 0..1 del fondo ya capturado (CAPTURING).


PERSON_COVERAGE = 0.03     # Más de 3 % del cuadro con persona = no está vacío.
EMPTY_SECONDS = 1.0        # Escena vacía este tiempo antes de capturar.
SAMPLE_SECONDS = 0.6       # Cuadros que se promedian (mediana) para el fondo.
SCENE_CHECK_SECONDS = 0.15
DETECT_SECONDS = 0.08
MASK_SECONDS = 0.05


class Engine:
    def __init__(self, detector: PoseDetector, segmenter: Segmenter,
                 on_teleport: Callable[[], None] = lambda: None,
                 style: Style = Style.ANIME, seed: int | None = None):
        self.detector = detector
        self.segmenter = segmenter
        self.on_teleport = on_teleport
        self.style = Style(style)
        self.hold = HoldTimer()
        self._rng = np.random.default_rng(seed)
        self.background: np.ndarray | None = None
        self._reset_capture(Phase.WAITING)
        self._mask: np.ndarray | None = None
        self._mask_at = float("-inf")
        self._detect_at = float("-inf")
        self._effect_started = 0.0
        self._streaks = make_streaks(self._rng)
        self._noise: np.ndarray | None = None
        self._manual = False

    # Órdenes de la interfaz -------------------------------------------------

    def recapture(self) -> None:
        self.background = None
        self.hold.reset()
        self._reset_capture(Phase.WAITING)

    def trigger(self) -> None:
        """Botón "Teletransportarme": lo mismo que completar el gesto (sin fondo no hace nada)."""
        if self.phase in (Phase.READY, Phase.HIDDEN):
            self._manual = True

    def set_style(self, style: Style | str) -> None:
        self.style = Style(style)

    # Cuadro a cuadro --------------------------------------------------------

    def process(self, frame: np.ndarray, now: float) -> tuple[np.ndarray, Status]:
        timestamp_ms = int(now * 1000)
        if self.phase in (Phase.WAITING, Phase.CAPTURING):
            return frame, self._calibrate(frame, now, timestamp_ms)

        if self.phase in (Phase.READY, Phase.HIDDEN):
            if now - self._detect_at >= DETECT_SECONDS:
                self._detect_at = now
                detected = self.detector.detect(frame, timestamp_ms)
                if self.hold.update(detected, now) == "triggered":
                    self._manual = True
            if self._manual:
                self._manual = False
                self._start_effect(now)

        output = frame
        if self.phase in (Phase.VANISHING, Phase.RETURNING):
            output = self._render_effect(frame, now, timestamp_ms)
        elif self.phase == Phase.HIDDEN:
            output = self.background.copy()
        elif self.style == Style.VFX and self.hold.started is not None:
            # El aura crece mientras se sostiene el gesto.
            self._refresh_mask(frame, now, timestamp_ms)
            output = render_aura(frame, self._mask, self._held(now), now, self._noise_for(frame))
        return output, self._status(now)

    # Internos ---------------------------------------------------------------

    def _reset_capture(self, phase: Phase) -> None:
        self.phase = phase
        self._samples: list[np.ndarray] = []
        self._empty_since: float | None = None
        self._capture_started = 0.0
        # Fuerza a revisar la escena en el siguiente cuadro.
        self._scene_at = float("-inf")
        self._scene_empty = False

    def _calibrate(self, frame: np.ndarray, now: float, timestamp_ms: int) -> Status:
        if now - self._scene_at >= SCENE_CHECK_SECONDS:
            self._scene_at = now
            mask = self.segmenter.mask(frame, timestamp_ms)
            self._scene_empty = float((mask > 0.5).mean()) < PERSON_COVERAGE
        if not self._scene_empty:
            # Alguien está en cuadro: se espera desde cero, pero sin perder el ritmo de revisión.
            checked_at = self._scene_at
            self._reset_capture(Phase.WAITING)
            self._scene_at = checked_at
            return Status(Phase.WAITING)
        if self.phase == Phase.WAITING:
            if self._empty_since is None:
                self._empty_since = now
            if now - self._empty_since < EMPTY_SECONDS:
                return Status(Phase.WAITING, empty_for=now - self._empty_since)
            self.phase = Phase.CAPTURING
            self._capture_started = now
        self._samples.append(frame.copy())
        done = (now - self._capture_started) / SAMPLE_SECONDS
        if done < 1.0:
            return Status(Phase.CAPTURING, capture=done)
        self.background = np.median(np.stack(self._samples[-20:]), axis=0).astype(np.uint8)
        self._samples = []
        self.phase = Phase.READY
        return Status(Phase.READY)

    def _start_effect(self, now: float) -> None:
        self.phase = Phase.VANISHING if self.phase == Phase.READY else Phase.RETURNING
        self._effect_started = now
        self._streaks = make_streaks(self._rng)
        self._mask_at = float("-inf")
        self.on_teleport()

    def _duration(self) -> float:
        return GOKU_SECONDS if self.style == Style.VFX else ANIME_SECONDS

    def _render_effect(self, frame: np.ndarray, now: float, timestamp_ms: int) -> np.ndarray:
        progress = min(1.0, (now - self._effect_started) / self._duration())
        # En vfx la silueta se congela cuando ya se cerró en una línea.
        frozen = (self.style == Style.VFX and self.phase == Phase.VANISHING
                  and progress >= SNAP_END)
        if not frozen:
            self._refresh_mask(frame, now, timestamp_ms)
        shown = progress if self.phase == Phase.VANISHING else 1.0 - progress
        if self.style == Style.VFX:
            output = render_teleport(frame, self.background, self._mask, shown, self._streaks,
                                     self._noise_for(frame))
        else:
            output = render_instant(frame, self.background, self._mask, shown, self._streaks)
        if progress >= 1.0:
            self.phase = Phase.HIDDEN if self.phase == Phase.VANISHING else Phase.READY
            self._mask = None
        return output

    def _refresh_mask(self, frame: np.ndarray, now: float, timestamp_ms: int) -> None:
        if self._mask is None or now - self._mask_at >= MASK_SECONDS:
            self._mask = self.segmenter.mask(frame, timestamp_ms)
            self._mask_at = now

    def _noise_for(self, frame: np.ndarray) -> np.ndarray:
        if self._noise is None or self._noise.shape != frame.shape[:2]:
            self._noise = make_aura_noise(self._rng, *frame.shape[:2])
        return self._noise

    def _held(self, now: float) -> float:
        if self.hold.started is None:
            return 0.0
        return 1.0 - self.hold.remaining(now) / self.hold.hold_seconds

    def _status(self, now: float) -> Status:
        return Status(self.phase, hold=self._held(now),
                      hands=self.detector.hands_seen, face=self.detector.face_seen)
