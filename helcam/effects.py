"""Efectos de teletransporte: estilo anime (tiras verticales) y estilo vfx."""

from __future__ import annotations

import math

import cv2
import numpy as np


GOKU_SECONDS = 0.85
# Como en el anime: unos cuantos cuadros y Goku ya no está.
ANIME_SECONDS = 0.38
# Visibilidad cuadro a cuadro durante el parpadeo "shun".
ANIME_BLINKS = (1.0, 0.3, 0.95, 0.0, 0.7, 0.15, 0.45, 0.0)
ANIME_BLINK_END = 0.6

# Fases del efecto vfx, en fracciones de GOKU_SECONDS.
CHARGE_END = 0.22
SNAP_END = 0.45
COLLAPSE_END = 0.55

STREAK_COLOR = (255, 246, 222)  # BGR: blanco apenas azulado
AURA_COLOR = (255, 228, 170)  # BGR: azul claro


def make_streaks(rng: np.random.Generator, count: int = 34) -> np.ndarray:
    """Columnas: x relativa, fase, largo relativo, grosor, velocidad."""
    return np.column_stack((
        rng.uniform(0.04, 0.96, count),
        rng.uniform(0.0, 1.0, count),
        rng.uniform(0.18, 0.55, count),
        rng.integers(1, 4, count),
        rng.uniform(1.4, 3.2, count),
    ))


def _ease(value: float) -> float:
    value = min(1.0, max(0.0, value))
    return value * value * (3.0 - 2.0 * value)


def _ease_out(value: float) -> float:
    value = min(1.0, max(0.0, value))
    return 1.0 - (1.0 - value) ** 3


def make_aura_noise(rng: np.random.Generator, height: int, width: int) -> np.ndarray:
    """Ruido suave del tamaño del cuadro: hace temblar el aura como una flama."""
    coarse = rng.random((max(2, height // 18), max(2, width // 12)), dtype=np.float32)
    return cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC).clip(0.0, 1.0)


def _person_box(mask: np.ndarray | None, width: int, height: int) -> tuple[int, int, int, int]:
    points = None if mask is None else cv2.findNonZero((mask > 0.3).astype(np.uint8))
    if points is None:
        return width // 3, height // 8, width - width // 3, height
    x, y, w, h = cv2.boundingRect(points)
    pad = max(24, w // 4)
    return max(0, x - pad), max(0, y - pad), min(width, x + w + pad), min(height, y + h + pad)


def _center(alpha: np.ndarray) -> tuple[float, float]:
    """Centro de masa de la silueta, en coordenadas del recuadro."""
    total = float(alpha.sum())
    height, width = alpha.shape
    if total <= 1e-3:
        return width * 0.5, height * 0.5
    return (float((alpha.sum(axis=0) * np.arange(width)).sum()) / total,
            float((alpha.sum(axis=1) * np.arange(height)).sum()) / total)


def _streak_layer(size: tuple[int, int], streaks: np.ndarray, progress: float) -> np.ndarray:
    """Líneas blancas que suben dentro del recuadro de la persona."""
    height, width = size
    layer = np.zeros((height, width), dtype=np.float32)
    span = height * 1.4
    for x, phase, length, thickness, speed in streaks:
        center = height * 1.2 - ((phase + progress * speed) % 1.0) * span
        half = length * height * 0.5
        column = int(x * (width - 1))
        cv2.line(layer, (column, int(center - half)), (column, int(center + half)),
                 1.0, int(thickness), cv2.LINE_AA)
    glow = cv2.GaussianBlur(layer, (0, 0), sigmaX=4.0, sigmaY=1.5)
    return np.clip(layer + glow * 1.6, 0.0, 1.0)


def _soft(alpha: np.ndarray, sigma: float) -> np.ndarray:
    """Desenfoque amplio a un cuarto de resolución: igual a la vista y mucho más rápido."""
    height, width = alpha.shape
    small = cv2.resize(alpha, (max(1, width // 4), max(1, height // 4)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigmaX=max(0.5, sigma / 4))
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)


def _blend(output: np.ndarray, color: np.ndarray | tuple, alpha: np.ndarray) -> np.ndarray:
    """output + (color - output) * alpha con OpenCV: con numpy costaba ~25 ms por llamada."""
    output = np.asarray(output, dtype=np.float32)
    alpha = cv2.merge((alpha, alpha, alpha)).astype(np.float32, copy=False)
    if isinstance(color, np.ndarray) and color.shape == output.shape:
        difference = cv2.subtract(output, color.astype(np.float32, copy=False))
    else:
        difference = cv2.subtract(output, (*map(float, color), 0.0))
    return cv2.subtract(output, cv2.multiply(difference, alpha))


def _apply_aura(region: np.ndarray, alpha: np.ndarray, strength: float, seconds: float,
                noise: np.ndarray | None, top: int, left: int) -> None:
    """Brillo que sube desde la silueta; strength 0..1 crece mientras se sostiene el gesto."""
    if strength <= 0.0:
        return
    height, width = alpha.shape
    glow = _soft(alpha, 12.0)
    # Fuera de la silueta: brillo que se desvanece; dentro: solo un borde fino.
    outer = np.clip(glow * 1.8, 0.0, 1.0) * (1.0 - alpha)
    if noise is not None:
        rows = (np.arange(top, top + height) + int(seconds * 220)) % noise.shape[0]
        outer *= 0.35 + 0.9 * noise[rows, left:left + width]
    rim = alpha * np.clip((1.0 - glow) * 2.5, 0.0, 1.0)
    region[:] = _blend(region, AURA_COLOR, np.clip((outer * 1.1 + rim * 0.6) * strength, 0.0, 0.9))


def render_aura(frame: np.ndarray, mask: np.ndarray | None, strength: float, seconds: float,
                noise: np.ndarray | None = None) -> np.ndarray:
    """Persona todavía visible, envuelta en el aura mientras tiene los dedos en la frente."""
    if mask is None or strength <= 0.0:
        return frame
    height, width = frame.shape[:2]
    left, top, right, bottom = _person_box(mask, width, height)
    output = frame.copy()
    region = output[top:bottom, left:right].astype(np.float32)
    _apply_aura(region, mask[top:bottom, left:right].astype(np.float32),
                min(1.0, strength), seconds, noise, top, left)
    output[top:bottom, left:right] = np.clip(region, 0, 255).astype(np.uint8)
    return output


def _shockwave(output: np.ndarray, center: tuple[float, float], radius: float,
               amplitude: float, glow: float) -> None:
    """Anillo que deforma el fondo, como aire desplazado de golpe."""
    if radius <= 1.0 or (amplitude <= 0.2 and glow <= 0.01):
        return
    height, width = output.shape[:2]
    thickness = float(np.clip(radius * 0.1, 8.0, 24.0))
    reach = int(radius + thickness * 2.5)
    cx, cy = center
    left, right = max(0, int(cx) - reach), min(width, int(cx) + reach + 1)
    top, bottom = max(0, int(cy) - reach), min(height, int(cy) + reach + 1)
    if left >= right or top >= bottom:
        return
    # Mapas a media resolución y luego ampliados: el anillo es suave.
    size = (right - left, bottom - top)
    ys, xs = np.mgrid[top:bottom:2, left:right:2].astype(np.float32)
    dx, dy = xs - cx, ys - cy
    distance = np.sqrt(dx * dx + dy * dy) + 1e-3
    ring = np.exp(-(((distance - radius) / thickness) ** 2))
    push = amplitude * ring * np.sin((distance - radius) / thickness * math.pi)
    map_x = cv2.resize(xs - dx / distance * push, size, interpolation=cv2.INTER_LINEAR)
    map_y = cv2.resize(ys - dy / distance * push, size, interpolation=cv2.INTER_LINEAR)
    ring = cv2.resize(ring, size, interpolation=cv2.INTER_LINEAR)
    patch = cv2.remap(output, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    output[top:bottom, left:right] = np.clip(
        _blend(patch.astype(np.float32), (255, 255, 255), ring * glow), 0, 255).astype(np.uint8)


def render_teleport(frame: np.ndarray, background: np.ndarray, mask: np.ndarray | None,
                    progress: float, streaks: np.ndarray,
                    noise: np.ndarray | None = None) -> np.ndarray:
    """0 = persona completa, 1 = desaparecida. Para reaparecer se usa 1 - progress."""
    if progress <= 0.0:
        if mask is None:
            return frame.copy()
        return _blend(background.astype(np.float32), frame.astype(np.float32),
                      mask).astype(np.uint8)
    if progress >= 1.0:
        return background.copy()

    height, width = frame.shape[:2]
    left, top, right, bottom = _person_box(mask, width, height)
    area = np.s_[top:bottom, left:right]
    output = background.copy()
    region = output[area].astype(np.float32)
    box_height, box_width = region.shape[:2]
    alpha = (np.zeros((box_height, box_width), np.float32) if mask is None
             else mask[area].astype(np.float32))
    person = frame[area].astype(np.float32)
    center_x, center_y = _center(alpha)
    seconds = progress * GOKU_SECONDS

    if progress < SNAP_END:
        if progress < CHARGE_END:
            # Parpadeo cada vez más fuerte mientras se concentra el ki.
            charge = progress / CHARGE_END
            blink = 0.5 + 0.5 * math.copysign(1.0, math.sin(progress * math.tau * 11.0))
            region[:] = _blend(region, person, alpha * (1.0 - 0.5 * charge * blink))
            _apply_aura(region, alpha, min(1.0, progress / 0.06), seconds, noise, top, left)
        else:
            squeeze = _ease((progress - CHARGE_END) / (SNAP_END - CHARGE_END))
            scale_x = max(0.015, 1.0 - squeeze)
            scale_y = 1.0 + 0.15 * squeeze
            matrix = np.float32(((scale_x, 0, center_x * (1 - scale_x)),
                                 (0, scale_y, center_y * (1 - scale_y))))
            person = cv2.warpAffine(person, matrix, (box_width, box_height),
                                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            squeezed = cv2.warpAffine(alpha, matrix, (box_width, box_height),
                                      flags=cv2.INTER_LINEAR)
            person = _blend(person, STREAK_COLOR,
                            np.full(squeezed.shape, squeeze * 0.8, np.float32))
            # Separación RGB: azul y rojo se deslizan a lados opuestos.
            shift = int(round(squeeze * 10))
            for channel, offset in ((0, shift), (1, 0), (2, -shift)):
                channel_alpha = np.roll(squeezed, offset, axis=1)
                channel_person = np.roll(person[:, :, channel], offset, axis=1)
                region[:, :, channel] = (region[:, :, channel] * (1.0 - channel_alpha)
                                         + channel_person * channel_alpha)
            halo = _soft(squeezed, 8.0)
            region[:] = _blend(region, AURA_COLOR, np.clip(halo * 1.5 - squeezed, 0.0, 0.8))

    # Destello: la línea vertical se cierra en un punto de luz.
    if SNAP_END - 0.05 <= progress < COLLAPSE_END + 0.08 and mask is not None:
        collapse = _ease((progress - SNAP_END) / (COLLAPSE_END - SNAP_END))
        fade = 1.0 - max(0.0, progress - COLLAPSE_END) / 0.08
        half = box_height * 0.55 * (1.0 - collapse)
        line = np.zeros((box_height, box_width), np.float32)
        x = int(center_x)
        cv2.line(line, (x, int(center_y - half)), (x, int(center_y + half)), 1.0, 3, cv2.LINE_AA)
        cv2.circle(line, (x, int(center_y)), max(2, int(6 + 10 * collapse)), 1.0, -1, cv2.LINE_AA)
        line = np.clip(line + _soft(line, 10.0) * 3.0, 0.0, 1.0)
        region[:] = _blend(region, (255, 255, 255), line * fade)

    # Líneas de velocidad: primero sobre la silueta, luego se quedan un instante donde estaba la persona.
    strength = min(1.0, max(0.0, (progress - 0.03) / 0.15))
    if progress < SNAP_END and mask is not None:
        coverage = _soft(cv2.dilate(mask[area], np.ones((15, 15), np.uint8)), 6.0)
    else:
        rows = np.linspace(0.0, 1.0, box_height, dtype=np.float32)
        coverage = np.repeat(np.minimum(1.0, np.minimum(rows, 1.0 - rows) * 5.0)[:, None],
                             box_width, axis=1)
        strength *= 1.0 - _ease((progress - SNAP_END) / (1.0 - SNAP_END))
    if strength > 0.0:
        streak = _streak_layer((box_height, box_width), streaks, progress)
        region[:] = _blend(region, STREAK_COLOR, streak * coverage * strength * 0.9)

    output[area] = np.clip(region, 0, 255).astype(np.uint8)
    if progress >= SNAP_END and mask is not None:
        wave = _ease_out((progress - SNAP_END) / (1.0 - SNAP_END))
        _shockwave(output, (left + center_x, top + center_y),
                   radius=wave * max(height, width) * 0.45,
                   amplitude=16.0 * (1.0 - wave), glow=0.35 * (1.0 - wave) ** 2)
    return output


STRIP_WIDTH = 6


def _tear(image: np.ndarray, alpha: np.ndarray, amount: float, phase: float,
          dropped: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """La silueta se parte en tiras verticales que se deslizan arriba y abajo.

    dropped (0..1) es la fracción de tiras que ya desaparecieron.
    """
    height, width = alpha.shape
    columns = np.arange(width, dtype=np.float32)
    strips = columns // STRIP_WIDTH
    direction = (strips % 2) * 2.0 - 1.0
    offset = amount * (0.65 * direction + 0.35 * np.sin(columns * 0.05 + phase))
    map_x = np.tile(columns, (height, 1))
    map_y = np.arange(height, dtype=np.float32)[:, None] - offset[None, :]
    torn = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    torn_alpha = cv2.remap(alpha, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    if dropped > 0.0:
        # Orden pseudoaleatorio pero estable: cada tira desaparece siempre en el mismo punto.
        threshold = np.modf(np.abs(np.sin(strips * 12.9898)) * 43758.5453)[0]
        torn_alpha *= (threshold >= dropped).astype(np.float32)[None, :]
    return torn, torn_alpha


def render_instant(frame: np.ndarray, background: np.ndarray, mask: np.ndarray | None,
                   progress: float, streaks: np.ndarray) -> np.ndarray:
    """Teletransporte estilo anime: parpadeo en tiras y luego un eco con líneas verticales.

    0 = persona completa, 1 = desaparecida. Para reaparecer se usa 1 - progress.
    """
    if progress <= 0.0:
        if mask is None:
            return frame.copy()
        return _blend(background.astype(np.float32), frame.astype(np.float32),
                      mask).astype(np.uint8)
    if progress >= 1.0 or mask is None:
        return background.copy()

    height, width = frame.shape[:2]
    left, top, right, bottom = _person_box(mask, width, height)
    area = np.s_[top:bottom, left:right]
    output = background.copy()
    region = output[area].astype(np.float32)
    alpha = mask[area].astype(np.float32)
    person = frame[area].astype(np.float32)

    if progress < ANIME_BLINK_END:
        step = progress / ANIME_BLINK_END
        visibility = ANIME_BLINKS[min(len(ANIME_BLINKS) - 1, int(step * len(ANIME_BLINKS)))]
        if visibility > 0.0:
            torn, torn_alpha = _tear(person, alpha, 2.0 + 18.0 * step, progress * 47.0,
                                     dropped=0.7 * step)
            # Entre más se desvanece, más se vuelve una silueta clara y plana.
            torn = _blend(torn, STREAK_COLOR, np.full(torn_alpha.shape, 0.45 * step, np.float32))
            region[:] = _blend(region, torn, torn_alpha * visibility)
    else:
        fade = 1.0 - _ease((progress - ANIME_BLINK_END) / (1.0 - ANIME_BLINK_END))
        # Eco transparente y muy rasgado donde estaba la persona.
        ghost, ghost_alpha = _tear(person, alpha, 24.0, progress * 47.0, dropped=0.6)
        ghost = _blend(ghost, STREAK_COLOR, np.full(ghost_alpha.shape, 0.6, np.float32))
        region[:] = _blend(region, ghost, ghost_alpha * 0.3 * fade)

    # Pocas líneas verticales finas, solo sobre la silueta.
    strength = min(1.0, progress / 0.2) * (1.0 - _ease((progress - 0.55) / 0.45))
    if strength > 0.0:
        coverage = _soft(cv2.dilate(alpha, np.ones((9, 9), np.uint8)), 5.0)
        streak = _streak_layer(alpha.shape, streaks[::2], progress * 2.0)
        region[:] = _blend(region, STREAK_COLOR, streak * coverage * strength * 0.75)

    output[area] = np.clip(region, 0, 255).astype(np.uint8)
    return output

