"""Cámaras de Linux (V4L2): webcams reales, la infrarroja y la virtual VanishCam."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

SYSFS = Path("/sys/class/video4linux")
VIRTUAL_LABEL = "VanishCam"


@dataclass(frozen=True)
class VideoDevice:
    index: int
    name: str
    virtual: bool
    infrared: bool

    @property
    def path(self) -> str:
        return f"/dev/video{self.index}"

    @property
    def label(self) -> str:
        # "HP HD Camera: HP HD Camera" -> "HP HD Camera"
        name = self.name.split(": ", 1)[-1]
        return f"{name} (infrarroja)" if self.infrared else name


def list_devices(sysfs: Path = SYSFS) -> list[VideoDevice]:
    """Un nodo por dispositivo: los nodos index=1 de UVC son metadatos, no video."""
    devices = []
    for node in sysfs.glob("video*"):
        match = re.fullmatch(r"video(\d+)", node.name)
        if not match:
            continue
        try:
            if (node / "index").read_text().strip() != "0":
                continue
            name = (node / "name").read_text().strip()
        except OSError:
            continue
        virtual = "/virtual/" in str((node / "device").resolve()) or name == VIRTUAL_LABEL
        infrared = bool(re.search(r"\bIR\b|infrared", name, re.IGNORECASE))
        devices.append(VideoDevice(int(match.group(1)), name, virtual, infrared))
    return sorted(devices, key=lambda device: device.index)


def webcams(sysfs: Path = SYSFS) -> list[VideoDevice]:
    """Cámaras reales; la infrarroja al final porque solo da imagen en grises."""
    cameras = [device for device in list_devices(sysfs) if not device.virtual]
    return sorted(cameras, key=lambda device: (device.infrared, device.index))


def virtual_device(sysfs: Path = SYSFS) -> VideoDevice | None:
    loopbacks = [device for device in list_devices(sysfs) if device.virtual]
    named = [device for device in loopbacks if device.name == VIRTUAL_LABEL]
    return (named or loopbacks or [None])[0]


class Webcam:
    def __init__(self, index: int, width: int = 1280, height: int = 720, fps: int = 30):
        self._capture = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not self._capture.isOpened():
            raise RuntimeError(f"No pude abrir /dev/video{index}. ¿La está usando otra app?")
        # MJPG permite 720p a 30 fps en webcams USB; YUYV suele quedarse en 10 fps.
        self._capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._capture.set(cv2.CAP_PROP_FPS, fps)
        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def read(self) -> np.ndarray:
        ok, frame = self._capture.read()
        if not ok:
            raise RuntimeError("La cámara dejó de mandar imagen.")
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        return frame

    def close(self) -> None:
        self._capture.release()


class VirtualCamera:
    """Salida a VanishCam (v4l2loopback): siempre derecha, sin espejo ni indicadores."""

    def __init__(self, device: str, width: int, height: int, fps: float = 30.0):
        import pyvirtualcam

        self._camera = pyvirtualcam.Camera(width=width, height=height, fps=fps,
                                           fmt=pyvirtualcam.PixelFormat.BGR,
                                           backend="v4l2loopback", device=device)
        self.device = device

    def send(self, frame: np.ndarray) -> None:
        if frame.shape[:2] != (self._camera.height, self._camera.width):
            frame = cv2.resize(frame, (self._camera.width, self._camera.height))
        self._camera.send(np.ascontiguousarray(frame))

    def close(self) -> None:
        self._camera.close()
