"""Sonido del teletransporte: el tuyo en ~/.config/helcam o uno sintético."""

from __future__ import annotations

import math
import os
import wave
from pathlib import Path

import numpy as np

from .paths import CACHE_DIR, CONFIG_DIR

SOUND_SUFFIXES = (".mp3", ".wav", ".ogg")
GENERATED_NAME = "teleport_sintetico.wav"


def teleport_sound_path(config_dir: Path = CONFIG_DIR, cache_dir: Path = CACHE_DIR) -> Path:
    """config_dir/teleport.<audio> (mayúsculas o no) gana; si no hay, se genera un "shun"."""
    if config_dir.is_dir():
        custom = sorted(path for path in config_dir.iterdir()
                        if path.stem.lower() == "teleport"
                        and path.suffix.lower() in SOUND_SUFFIXES and path.is_file())
        if custom:
            return custom[0]
    generated = cache_dir / GENERATED_NAME
    if not generated.is_file():
        write_teleport_sound(generated)
    return generated


def write_teleport_sound(path: Path, rate: int = 44100) -> None:
    duration = 0.42
    t = np.arange(int(rate * duration)) / rate
    # Silbido que baja, soplo corto y golpe grave.
    frequency = 600.0 + 2600.0 * np.exp(-t / 0.05)
    chirp = np.sin(math.tau * np.cumsum(frequency) / rate) * np.exp(-t / 0.09)
    rng = np.random.default_rng(4242)
    noise = rng.uniform(-1.0, 1.0, t.size)
    noise = (noise - np.convolve(noise, np.ones(8) / 8, mode="same")) * np.exp(-t / 0.04)
    thump = np.sin(math.tau * 110.0 * t) * np.exp(-t / 0.07)
    attack = np.minimum(1.0, t / 0.004)
    signal = (chirp * 0.6 + noise * 0.35 + thump * 0.45) * attack
    signal *= 0.85 / max(1e-6, float(np.abs(signal).max()))
    samples = (signal * 32767).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(np.repeat(samples, 2).tobytes())


class TeleportSound:
    """Decodifica el audio una vez y lo reproduce sin bloquear el video."""

    def __init__(self, path: Path):
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        from pygame import mixer

        self.path = path
        self._mixer = mixer
        if mixer.get_init() is None:
            mixer.init(frequency=44100, size=-16, channels=2, buffer=256)
        self._sound = mixer.Sound(str(path))

    def play(self) -> None:
        self._sound.stop()
        self._sound.play()

    def close(self) -> None:
        self._mixer.quit()
