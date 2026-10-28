import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from helcam.app import MainWindow, Toggle
from helcam.engine import Phase, Status, Style
from helcam.worker import Options, to_qimage


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


class FakeWorker:
    def __init__(self):
        self.sent = []

    def send(self, command, value=None):
        self.sent.append((command, value))


class FakeSettings:
    def setValue(self, key, value):
        pass


@pytest.fixture
def window(app, tmp_path, monkeypatch):
    # Que los tests no toquen tu configuración guardada.
    monkeypatch.setattr("helcam.app.QSettings", lambda *args: FakeSettings())
    window = MainWindow(Options(style=Style.ANIME), start_worker=False)
    window.worker = FakeWorker()
    return window


def test_buttons_send_commands_to_the_video_thread(window):
    window.teleport_button.setEnabled(True)
    window.teleport_button.click()
    vfx = next(b for b in window.style_group.buttons() if b.text() == "VFX")
    vfx.click()
    window.sound_toggle.click()
    assert window.worker.sent == [("trigger", None), ("style", "vfx"), ("sound", False)]


def test_status_updates_the_panel(window):
    frame = to_qimage(np.zeros((90, 160, 3), np.uint8))
    window._on_frame(frame, Status(Phase.HIDDEN, hands=1, face=True))
    assert window.phase_label.text() == "Invisible"
    assert window.teleport_button.text() == "Regresar"
    assert window.teleport_button.isEnabled()
    window._on_frame(frame, Status(Phase.WAITING))
    assert not window.teleport_button.isEnabled()


def test_output_state_follows_the_worker_without_echo(window):
    window._on_output("")
    assert not window.virtual_toggle.isChecked()
    window._on_output("/dev/video10")
    assert window.virtual_toggle.isChecked()
    # El cambio vino del hilo: no se le vuelve a mandar.
    assert window.worker.sent == []


def test_toggle_flips(app):
    toggle = Toggle()
    toggle.click()
    assert toggle.isChecked()
