"""Hilo de video: cámara -> motor -> VanishCam, y vista previa para la ventana."""

from __future__ import annotations

import queue
import time
from dataclasses import dataclass, replace

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from .camera import VirtualCamera, Webcam, virtual_device
from .engine import Engine, Style
from .sound import TeleportSound, teleport_sound_path
from .vision import GokuPoseDetector, PersonSegmenter, ensure_models


@dataclass(frozen=True)
class Options:
    camera: int = 0
    style: Style = Style.ANIME
    mirror: bool = True
    virtual: bool = True
    sound: bool = True


def to_qimage(frame_bgr: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    return QImage(rgb.data, width, height, rgb.strides[0], QImage.Format.Format_RGB888).copy()


class VideoWorker(QThread):
    frame_ready = Signal(QImage, object)   # vista previa, engine.Status
    loading = Signal(str)                  # texto mientras arranca
    output_changed = Signal(str)           # dispositivo virtual activo, "" si está apagado
    problem = Signal(str)                  # errores que la ventana debe mostrar

    def __init__(self, options: Options):
        super().__init__()
        self.options = options
        self._commands: queue.SimpleQueue[tuple[str, object]] = queue.SimpleQueue()
        self._running = True

    # Llamadas seguras desde el hilo de la interfaz.
    def send(self, command: str, value: object = None) -> None:
        self._commands.put((command, value))

    def stop(self) -> None:
        self._running = False
        self.wait(3000)

    def run(self) -> None:
        detector = segmenter = sound = webcam = output = None
        try:
            self.loading.emit("Descargando modelos (solo la primera vez)…")
            models = ensure_models(progress=lambda name, part: self.loading.emit(
                f"Descargando {name}… {part:.0%}"))
            self.loading.emit("Cargando modelos…")
            detector = GokuPoseDetector(models["hand_landmarker.task"], models["face_landmarker.task"])
            segmenter = PersonSegmenter(models["selfie_segmenter_landscape.tflite"])
            try:
                sound = TeleportSound(teleport_sound_path())
            except Exception as error:  # Sin audio la app sigue funcionando.
                self.problem.emit(f"Sin sonido: {error}")
            engine = Engine(detector, segmenter, style=self.options.style,
                            on_teleport=lambda: sound and self.options.sound and sound.play())
            self.loading.emit("Abriendo la cámara…")
            webcam = Webcam(self.options.camera)
            want_output = self.options.virtual

            while self._running:
                while not self._commands.empty():
                    command, value = self._commands.get()
                    if command == "recapture":
                        engine.recapture()
                    elif command == "trigger":
                        engine.trigger()
                    elif command == "style":
                        self.options = replace(self.options, style=Style(value))
                        engine.set_style(value)
                    elif command == "mirror":
                        self.options = replace(self.options, mirror=bool(value))
                    elif command == "sound":
                        self.options = replace(self.options, sound=bool(value))
                    elif command == "camera":
                        self.options = replace(self.options, camera=int(value))
                        webcam.close()
                        webcam = Webcam(self.options.camera)
                        engine.recapture()
                    elif command == "virtual":
                        want_output = bool(value)
                        if not want_output and output is not None:
                            output.close()
                            output = None
                            self.output_changed.emit("")

                frame = webcam.read()
                result, status = engine.process(frame, time.monotonic())

                if want_output and output is None:
                    output = self._open_output(result)
                    if output is None:
                        want_output = False
                if output is not None:
                    output.send(result)

                preview = cv2.flip(result, 1) if self.options.mirror else result
                self.frame_ready.emit(to_qimage(preview), status)
        except Exception as error:
            self.problem.emit(str(error))
        finally:
            for resource in (output, webcam, sound, segmenter, detector):
                if resource is not None:
                    try:
                        resource.close()
                    except Exception:
                        pass

    def _open_output(self, frame: np.ndarray) -> VirtualCamera | None:
        device = virtual_device()
        if device is None:
            self.problem.emit(
                "No encontré la cámara virtual. Cárgala con:\n"
                "sudo modprobe v4l2loopback devices=1 video_nr=10 "
                "card_label=VanishCam exclusive_caps=1")
            self.output_changed.emit("")
            return None
        try:
            height, width = frame.shape[:2]
            camera = VirtualCamera(device.path, width, height)
        except Exception as error:
            if "not a video output device" in str(error):
                # Con exclusive_caps=1 solo una app puede mandar video a la vez.
                self.problem.emit(
                    f"Otra app ya está mandando video a {device.label} ({device.path}).\n"
                    "Ciérrala y vuelve a encender «Cámara virtual».")
            else:
                self.problem.emit(f"No pude usar {device.path}: {error}")
            self.output_changed.emit("")
            return None
        self.output_changed.emit(device.path)
        return camera
