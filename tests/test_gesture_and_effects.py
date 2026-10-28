import math
from types import SimpleNamespace

import numpy as np

from helcam.effects import make_aura_noise, make_streaks, render_aura, render_instant, render_teleport
from helcam.gesture import (
    BROW_CENTER, FACE_LEFT, FACE_RIGHT, FOREHEAD_TOP, HoldTimer, two_finger_pose,
    two_fingers_on_forehead,
)
import unittest


def hand_points(index=True, middle=True, ring=False, pinky=False, spread=0.0):
    """Mano vertical en unidades arbitrarias, dedos hacia arriba (y negativa)."""
    points = np.zeros((21, 3))
    points[0] = (0.0, 0.0, 0.0)
    for base, x, opened, lean in zip(
        (5, 9, 13, 17), (-0.15, 0.05, 0.25, 0.45),
        (index, middle, ring, pinky), (-spread, spread, 0.0, 0.0),
    ):
        if opened:
            points[base : base + 4] = [(x + lean * k, -1.0 - 0.6 * k, 0.0) for k in range(4)]
        else:
            points[base : base + 4] = [
                (x, -1.0, 0.0), (x, -1.45, 0.2), (x, -1.25, 0.5), (x, -0.95, 0.3),
            ]
    return points


def landmarks(points):
    return [SimpleNamespace(x=x, y=y, z=z) for x, y, z in points]


def face_landmarks(width, height):
    """Cara de 160 px de ancho centrada en (480, 250); frente entre y=150 y y=215."""
    points = [SimpleNamespace(x=480 / width, y=250 / height, z=0.0) for _ in range(468)]
    points[FOREHEAD_TOP] = SimpleNamespace(x=480 / width, y=150 / height, z=0.0)
    points[BROW_CENTER] = SimpleNamespace(x=480 / width, y=215 / height, z=0.0)
    points[FACE_LEFT] = SimpleNamespace(x=400 / width, y=240 / height, z=0.0)
    points[FACE_RIGHT] = SimpleNamespace(x=560 / width, y=240 / height, z=0.0)
    return points


def hand_at(points, tip_x, tip_y, width, height, scale=40.0):
    """Mueve la mano en píxeles para que la punta del medio caiga en (tip_x, tip_y)."""
    shifted = points[:, :2] * scale
    shifted += np.array((tip_x, tip_y)) - shifted[12]
    return landmarks([(x / width, y / height, 0.0) for x, y in shifted])


class GokuPoseTests(unittest.TestCase):
    def test_two_joined_straight_fingers(self):
        self.assertTrue(two_finger_pose(hand_points()))

    def test_rejects_v_sign_open_hand_and_single_finger(self):
        self.assertFalse(two_finger_pose(hand_points(spread=0.35)))
        self.assertFalse(two_finger_pose(hand_points(ring=True, pinky=True)))
        self.assertFalse(two_finger_pose(hand_points(middle=False)))

    def test_pose_survives_rotation(self):
        for angle in (0.4, 1.2, math.pi / 2, 2.5):
            c, s = math.cos(angle), math.sin(angle)
            rotation = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))
            self.assertTrue(two_finger_pose(hand_points() @ rotation.T))

    def test_fingers_must_touch_the_forehead(self):
        width, height = 960, 540
        face = face_landmarks(width, height)
        pose = hand_points()
        on_forehead = hand_at(pose, 482, 185, width, height)
        on_chin = hand_at(pose, 482, 330, width, height)
        self.assertTrue(two_fingers_on_forehead([on_forehead], [], face, width, height))
        self.assertFalse(two_fingers_on_forehead([on_chin], [], face, width, height))
        self.assertFalse(two_fingers_on_forehead([on_forehead], [], None, width, height))
        open_hand = hand_at(hand_points(ring=True, pinky=True), 482, 185, width, height)
        self.assertFalse(two_fingers_on_forehead([open_hand], [], face, width, height))

    def test_world_landmarks_decide_the_finger_shape(self):
        width, height = 960, 540
        face = face_landmarks(width, height)
        image_hand = hand_at(hand_points(), 482, 185, width, height)
        open_world = landmarks(hand_points(ring=True, pinky=True))
        self.assertFalse(two_fingers_on_forehead([image_hand], [open_world], face, width, height))


class HoldTimerTests(unittest.TestCase):
    def test_triggers_after_one_second_then_needs_release(self):
        hold = HoldTimer()
        self.assertEqual(hold.update(True, 0.0), "started")
        self.assertIsNone(hold.update(True, 0.6))
        self.assertEqual(hold.update(True, 1.0), "triggered")
        # Mano todavía en la frente: no hay segundo disparo.
        for now in (1.1, 2.0, 3.0):
            self.assertIsNone(hold.update(True, now))
        self.assertIsNone(hold.update(False, 3.1))
        self.assertIsNone(hold.update(False, 3.6))
        self.assertEqual(hold.update(True, 3.7), "started")
        self.assertEqual(hold.update(True, 4.75), "triggered")

    def test_breaking_the_pose_restarts_the_countdown(self):
        hold = HoldTimer()
        hold.update(True, 0.0)
        hold.update(True, 0.7)
        self.assertIsNone(hold.update(False, 0.8))  # Corte breve del tracking.
        self.assertIsNone(hold.update(True, 0.9))
        self.assertIsNone(hold.update(False, 1.3))
        self.assertEqual(hold.update(False, 1.5), "lost")
        self.assertEqual(hold.update(True, 1.6), "started")
        self.assertIsNone(hold.update(True, 2.5))
        self.assertEqual(hold.update(True, 2.65), "triggered")


class TeleportEffectTests(unittest.TestCase):
    def setUp(self):
        self.background = np.full((120, 160, 3), 30, np.uint8)
        self.frame = np.full((120, 160, 3), 200, np.uint8)
        self.mask = np.zeros((120, 160), np.float32)
        self.mask[20:120, 60:100] = 1.0
        self.streaks = make_streaks(np.random.default_rng(3))

    def render(self, progress):
        return render_teleport(self.frame, self.background, self.mask, progress, self.streaks)

    def test_endpoints_show_person_and_background(self):
        np.testing.assert_array_equal(self.render(0.0)[50:100, 65:95], self.frame[50:100, 65:95])
        np.testing.assert_array_equal(self.render(1.0), self.background)

    def test_silhouette_squeezes_into_a_line(self):
        no_streaks = np.zeros((0, 5))
        squeezed = render_teleport(self.frame, self.background, self.mask, 0.42, no_streaks)
        # El centro sigue brillante; los bordes de la silueta vuelven a ser fondo.
        self.assertGreater(squeezed[70, 80].mean(), 150)
        self.assertLess(squeezed[70, 62].mean(), 120)

    def test_streaks_linger_only_where_the_person_was(self):
        linger = self.render(0.7)
        self.assertGreater(linger[:, 55:105].max(), 100)
        # Lejos de la silueta solo queda el último reflejo de la onda expansiva.
        far = np.abs(linger[:, :30].astype(int) - self.background[:, :30])
        self.assertLessEqual(far.max(), 6)

    def test_aura_glows_around_the_silhouette_not_inside(self):
        frame = self.background.copy()
        frame[self.mask > 0] = 200
        noise = make_aura_noise(np.random.default_rng(5), 120, 160)
        aura = render_aura(frame, self.mask, 1.0, 0.0, noise)
        self.assertGreater(aura[60, 50:58].astype(int).mean(), 100)  # Fondo 30 iluminado.
        # Dentro de la silueta solo un leve tono del borde.
        self.assertLessEqual(np.abs(aura[60:100, 75:85].astype(int) - 200).max(), 15)
        self.assertIs(render_aura(frame, self.mask, 0.0, 0.0, noise), frame)

    def test_anime_style_flickers_then_leaves_only_background(self):
        visible = [np.abs(render_instant(self.frame, self.background, self.mask, p, self.streaks)
                          .astype(int) - 30)[30:110, 60:100].mean()
                   for p in np.linspace(0.02, 0.58, 16)]
        # Parpadeo: la silueta desaparece y reaparece varias veces antes del eco.
        drops = sum(1 for a, b in zip(visible, visible[1:]) if b < a * 0.5)
        self.assertGreaterEqual(drops, 2)
        np.testing.assert_array_equal(
            render_instant(self.frame, self.background, self.mask, 1.0, self.streaks),
            self.background)
        np.testing.assert_array_equal(
            render_instant(self.frame, self.background, self.mask, 0.0, self.streaks)[50:100, 65:95],
            self.frame[50:100, 65:95])

    def test_works_without_mask(self):
        for progress in (0.2, 0.5, 0.8):
            output = render_teleport(self.frame, self.background, None, progress, self.streaks)
            self.assertEqual(output.shape, self.frame.shape)
