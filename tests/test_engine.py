import numpy as np
import pytest

from helcam.engine import EMPTY_SECONDS, SAMPLE_SECONDS, Engine, Phase, Style

EMPTY, PERSON = 50, 200
STEP = 1 / 30


class FakeScene:
    """Segmentador y detector falsos que leen el mismo "mundo"."""

    def __init__(self):
        self.person = True
        self.pose = False
        self.hands_seen = 0
        self.face_seen = False

    def mask(self, frame, timestamp_ms):
        # La persona ocupa el centro, como frente a una webcam.
        mask = np.zeros(frame.shape[:2], np.float32)
        if self.person:
            mask[12:, 20:44] = 1.0
        return mask

    def detect(self, frame, timestamp_ms):
        self.hands_seen = int(self.pose)
        self.face_seen = self.person
        return self.person and self.pose


def frame_for(scene):
    return np.full((48, 64, 3), PERSON if scene.person else EMPTY, np.uint8)


class Runner:
    def __init__(self, style=Style.ANIME):
        self.scene = FakeScene()
        self.teleports = 0
        self.engine = Engine(self.scene, self.scene, on_teleport=self._count, style=style, seed=1)
        self.now = 0.0
        self.output = None
        self.status = None

    def _count(self):
        self.teleports += 1

    def run(self, seconds):
        for _ in range(round(seconds / STEP)):
            self.now += STEP
            self.output, self.status = self.engine.process(frame_for(self.scene), self.now)
        return self.status

    def calibrated(self):
        self.scene.person = False
        self.run(EMPTY_SECONDS + SAMPLE_SECONDS + 0.4)
        assert self.engine.phase == Phase.READY
        self.scene.person = True
        self.run(0.2)
        return self


def test_waits_while_someone_is_in_view():
    runner = Runner()
    status = runner.run(3.0)
    assert status.phase == Phase.WAITING
    assert runner.engine.background is None
    # La vista previa muestra la cámara tal cual mientras espera.
    assert np.all(runner.output == PERSON)


def test_background_is_captured_once_the_scene_is_empty():
    runner = Runner()
    runner.run(1.0)
    runner.scene.person = False
    status = runner.run(EMPTY_SECONDS * 0.5)
    assert status.phase == Phase.WAITING and status.empty_for > 0
    status = runner.run(EMPTY_SECONDS * 0.5 + SAMPLE_SECONDS * 0.5)
    assert status.phase == Phase.CAPTURING and 0 < status.capture < 1
    runner.run(SAMPLE_SECONDS)
    assert runner.engine.phase == Phase.READY
    assert np.all(runner.engine.background == EMPTY)


def test_walking_in_during_capture_starts_over():
    runner = Runner()
    runner.scene.person = False
    runner.run(EMPTY_SECONDS + SAMPLE_SECONDS * 0.4)
    assert runner.engine.phase == Phase.CAPTURING
    runner.scene.person = True
    assert runner.run(0.3).phase == Phase.WAITING
    assert runner.engine.background is None


@pytest.mark.parametrize("style", list(Style))
def test_gesture_vanishes_and_brings_you_back(style):
    runner = Runner(style).calibrated()
    runner.scene.pose = True
    status = runner.run(0.5)
    assert status.phase == Phase.READY and 0.3 < status.hold < 0.7
    # La detección corre cada 80 ms: el disparo llega poco después del segundo.
    runner.run(0.75)
    assert runner.engine.phase == Phase.VANISHING
    runner.run(1.0)
    assert runner.engine.phase == Phase.HIDDEN
    assert np.all(runner.output == EMPTY)            # Solo el fondo.
    # Con la mano todavía en la frente no vuelve: hay que bajarla primero.
    runner.run(1.5)
    assert runner.engine.phase == Phase.HIDDEN
    runner.scene.pose = False
    runner.run(0.7)
    runner.scene.pose = True
    runner.run(1.2)
    assert runner.engine.phase == Phase.RETURNING
    runner.scene.pose = False
    runner.run(1.0)
    assert runner.engine.phase == Phase.READY
    assert np.all(runner.output == PERSON)
    assert runner.teleports == 2


def test_button_teleports_without_the_gesture():
    runner = Runner()
    runner.engine.trigger()                  # Sin fondo todavía: se ignora.
    runner.run(0.2)
    assert runner.engine.phase == Phase.WAITING
    runner.calibrated()
    runner.engine.trigger()
    runner.run(0.6)
    assert runner.engine.phase == Phase.HIDDEN
    runner.engine.trigger()
    runner.run(0.6)
    assert runner.engine.phase == Phase.READY


def test_vfx_shows_the_aura_while_holding_and_anime_does_not():
    for style, changes in ((Style.VFX, True), (Style.ANIME, False)):
        runner = Runner(style).calibrated()
        runner.scene.pose = True
        runner.run(0.6)
        assert runner.engine.phase == Phase.READY
        assert bool(np.any(runner.output != PERSON)) is changes


def test_recapture_goes_back_to_waiting():
    runner = Runner().calibrated()
    runner.engine.recapture()
    assert runner.run(0.2).phase == Phase.WAITING
    assert runner.engine.background is None
