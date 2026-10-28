"""Ventana de HelCam: vista previa con indicadores y panel de controles."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from PySide6.QtCore import (
    Property, QEasingCurve, QPropertyAnimation, QRectF, QSettings, QSize, Qt, QTimer,
)
from PySide6.QtGui import (
    QColor, QFont, QIcon, QImage, QKeySequence, QPainter, QPainterPath, QPen, QShortcut,
)
from PySide6.QtWidgets import (
    QAbstractButton, QApplication, QButtonGroup, QComboBox, QFrame, QHBoxLayout, QLabel,
    QMainWindow, QMessageBox, QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget,
)

from .camera import webcams
from .engine import EMPTY_SECONDS, Phase, Status, Style
from .worker import Options, VideoWorker

ICON = Path(__file__).resolve().parent / "assets" / "icon.svg"

BG = "#0d0f14"
PANEL = "#12161e"
CARD = "#181d27"
BORDER = "#252c3a"
TEXT = "#e9edf5"
MUTED = "#8a93a6"
ORANGE = "#ff8a1f"
BLUE = "#6cc6ff"
GREEN = "#3ddc84"
RED = "#ff4d5e"
VIOLET = "#b48cff"

PHASES = {
    Phase.WAITING: ("Sal del cuadro", ORANGE),
    Phase.CAPTURING: ("Capturando fondo", BLUE),
    Phase.READY: ("Listo", GREEN),
    Phase.VANISHING: ("Teletransportando", BLUE),
    Phase.RETURNING: ("Regresando", BLUE),
    Phase.HIDDEN: ("Invisible", VIOLET),
}

STYLESHEET = f"""
QMainWindow, QWidget#root {{ background: {BG}; }}
QWidget {{ color: {TEXT}; font-size: 13px; }}
QFrame#sidebar {{ background: {PANEL}; border-left: 1px solid {BORDER}; }}
QFrame#card {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 14px; }}
QLabel#title {{ font-size: 20px; font-weight: 700; }}
QLabel#subtitle, QLabel#hint {{ color: {MUTED}; }}
QLabel#cardTitle {{ color: {MUTED}; font-size: 11px; font-weight: 600; letter-spacing: 1px; }}
QLabel#phase {{ font-size: 18px; font-weight: 700; }}
QPushButton {{
    background: {CARD}; border: 1px solid {BORDER}; border-radius: 10px;
    padding: 9px 14px; font-weight: 600;
}}
QPushButton:hover {{ border-color: #3a4357; }}
QPushButton:pressed {{ background: #202634; }}
QPushButton#primary {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #ffa04a, stop:1 {ORANGE});
    color: #1b0f02; border: none; font-size: 14px; padding: 12px 14px;
}}
QPushButton#primary:hover {{ background: #ffa04a; }}
QPushButton#primary:disabled {{ background: #3a3027; color: #8a7a68; }}
QPushButton#segment {{
    border-radius: 8px; padding: 0; min-height: 32px; background: transparent; border: none;
    color: {MUTED};
}}
QPushButton#segment:checked {{ background: #262d3c; color: {TEXT}; }}
QFrame#segments {{ background: {BG}; border: 1px solid {BORDER}; border-radius: 10px; }}
QScrollArea, QScrollArea > QWidget > QWidget {{ background: transparent; border: none; }}
QScrollBar:vertical {{ background: transparent; width: 8px; margin: 4px; }}
QScrollBar::handle:vertical {{ background: #2b3243; border-radius: 4px; min-height: 30px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QComboBox {{
    background: {BG}; border: 1px solid {BORDER}; border-radius: 10px; padding: 8px 10px;
}}
QComboBox::drop-down {{ border: none; width: 24px; }}
QComboBox QAbstractItemView {{
    background: {CARD}; border: 1px solid {BORDER}; selection-background-color: #262d3c;
}}
"""


class Toggle(QAbstractButton):
    """Interruptor tipo iOS con animación."""

    def __init__(self, checked: bool = False, parent: QWidget | None = None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setChecked(checked)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._offset = 1.0 if checked else 0.0
        self._animation = QPropertyAnimation(self, b"offset", self)
        self._animation.setDuration(140)
        self._animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.toggled.connect(self._animate)

    def sizeHint(self) -> QSize:
        return QSize(44, 24)

    def set_quietly(self, checked: bool) -> None:
        """Refleja un cambio que vino del hilo de video, sin volver a mandarlo."""
        self.blockSignals(True)
        self.setChecked(checked)
        self.blockSignals(False)
        self._animate(checked)

    def _animate(self, checked: bool) -> None:
        self._animation.stop()
        self._animation.setEndValue(1.0 if checked else 0.0)
        self._animation.start()

    def _get_offset(self) -> float:
        return self._offset

    def _set_offset(self, value: float) -> None:
        self._offset = value
        self.update()

    offset = Property(float, _get_offset, _set_offset)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        track = QColor(ORANGE) if self.isChecked() else QColor("#2b3243")
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(track)
        painter.drawRoundedRect(QRectF(0, 2, 44, 20), 10, 10)
        painter.setBrush(QColor("#ffffff"))
        painter.drawEllipse(QRectF(3 + self._offset * 20, 4, 16, 16))


class VideoView(QWidget):
    """Vista previa: los indicadores solo se ven aquí, nunca en la cámara virtual."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setMinimumSize(480, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._image: QImage | None = None
        self._status = Status(Phase.WAITING)
        self._loading = "Iniciando…"
        self._live = ""
        self._spin = 0.0
        self._timer = QTimer(self, interval=33, timeout=self._tick)
        self._timer.start()

    def set_frame(self, image: QImage, status: Status) -> None:
        self._image = image
        self._status = status
        self._loading = ""
        self.update()

    def set_loading(self, text: str) -> None:
        self._loading = text
        self.update()

    def set_live(self, device: str) -> None:
        self._live = device
        self.update()

    def _tick(self) -> None:
        self._spin = (self._spin + 0.12) % math.tau
        if self._image is None or self._status.phase in (Phase.WAITING, Phase.CAPTURING):
            self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        area = QRectF(self.rect()).adjusted(20, 20, -20, -20)

        if self._image is None:
            self._draw_frame_box(painter, area)
            self._draw_spinner(painter, area.center().x(), area.center().y() - 18, 18)
            self._text(painter, QRectF(area.left(), area.center().y() + 14, area.width(), 30),
                       self._loading, 14, MUTED)
            return

        scale = min(area.width() / self._image.width(), area.height() / self._image.height())
        width, height = self._image.width() * scale, self._image.height() * scale
        target = QRectF(area.center().x() - width / 2, area.center().y() - height / 2, width, height)
        clip = QPainterPath()
        clip.addRoundedRect(target, 18, 18)
        painter.save()
        painter.setClipPath(clip)
        painter.drawImage(target, self._image)
        status = self._status
        if status.phase in (Phase.WAITING, Phase.CAPTURING):
            painter.fillRect(target, QColor(8, 10, 14, 150))
        painter.restore()
        painter.setPen(QPen(QColor(BORDER), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(target, 18, 18)

        label, color = PHASES[status.phase]
        self._pill(painter, target.left() + 16, target.top() + 16, label, color)
        if self._live:
            self._pill(painter, None, target.top() + 16, "EN VIVO · VanishCam", RED,
                       right=target.right() - 16)

        if status.phase == Phase.WAITING:
            progress = min(1.0, status.empty_for / EMPTY_SECONDS)
            sub = ("Quédate fuera un momento…" if progress > 0
                   else "El fondo se captura solo cuando no haya nadie")
            self._center_card(painter, target, "Sal del cuadro", sub, progress, ORANGE)
        elif status.phase == Phase.CAPTURING:
            self._center_card(painter, target, "Capturando fondo", "No muevas la cámara",
                              status.capture, BLUE)
        elif status.hold > 0 and status.phase in (Phase.READY, Phase.HIDDEN):
            title = "Concentrando ki…" if status.phase == Phase.READY else "Regresando…"
            self._hold_ring(painter, target, status.hold, title)
        elif status.phase == Phase.HIDDEN:
            self._pill(painter, None, target.bottom() - 46,
                       "Eres invisible · dos dedos en la frente para volver", VIOLET,
                       center=target.center().x())

    # Dibujo -----------------------------------------------------------------

    def _draw_frame_box(self, painter: QPainter, area: QRectF) -> None:
        painter.setPen(QPen(QColor(BORDER), 1))
        painter.setBrush(QColor(CARD))
        painter.drawRoundedRect(area, 18, 18)

    def _draw_spinner(self, painter: QPainter, x: float, y: float, radius: float) -> None:
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(BORDER), 3))
        painter.drawEllipse(QRectF(x - radius, y - radius, radius * 2, radius * 2))
        pen = QPen(QColor(ORANGE), 3)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawArc(QRectF(x - radius, y - radius, radius * 2, radius * 2),
                        int(-math.degrees(self._spin) * 16), 100 * 16)

    def _text(self, painter: QPainter, rect: QRectF, text: str, size: int, color: str,
              bold: bool = False) -> None:
        font = QFont(self.font())
        font.setPixelSize(size)
        font.setBold(bold)
        painter.setFont(font)
        painter.setPen(QColor(color))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

    def _pill(self, painter: QPainter, left: float | None, top: float, text: str, color: str,
              right: float | None = None, center: float | None = None) -> None:
        font = QFont(self.font())
        font.setPixelSize(12)
        font.setBold(True)
        painter.setFont(font)
        width = painter.fontMetrics().horizontalAdvance(text) + 38
        if right is not None:
            left = right - width
        elif center is not None:
            left = center - width / 2
        rect = QRectF(left, top, width, 28)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(10, 12, 18, 190))
        painter.drawRoundedRect(rect, 14, 14)
        painter.setBrush(QColor(color))
        painter.drawEllipse(QRectF(rect.left() + 12, rect.center().y() - 4, 8, 8))
        painter.setPen(QColor(TEXT))
        painter.drawText(rect.adjusted(26, 0, -10, 0), Qt.AlignmentFlag.AlignVCenter, text)

    def _center_card(self, painter: QPainter, target: QRectF, title: str, subtitle: str,
                     progress: float, color: str) -> None:
        card = QRectF(0, 0, min(360, target.width() - 40), 132)
        card.moveCenter(target.center())
        painter.setPen(QPen(QColor(BORDER), 1))
        painter.setBrush(QColor(18, 22, 30, 235))
        painter.drawRoundedRect(card, 16, 16)
        self._text(painter, QRectF(card.left(), card.top() + 22, card.width(), 30), title, 20,
                   TEXT, bold=True)
        self._text(painter, QRectF(card.left(), card.top() + 54, card.width(), 24), subtitle, 13,
                   MUTED)
        bar = QRectF(card.left() + 28, card.bottom() - 34, card.width() - 56, 6)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#262d3c"))
        painter.drawRoundedRect(bar, 3, 3)
        if progress > 0:
            painter.setBrush(QColor(color))
            painter.drawRoundedRect(QRectF(bar.left(), bar.top(), bar.width() * progress, 6), 3, 3)

    def _hold_ring(self, painter: QPainter, target: QRectF, progress: float, title: str) -> None:
        radius = 34
        center_x, center_y = target.center().x(), target.bottom() - 70
        box = QRectF(center_x - 150, center_y - radius - 14, 300, radius * 2 + 28)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(10, 12, 18, 200))
        painter.drawRoundedRect(box, 22, 22)
        ring = QRectF(box.left() + 14, center_y - radius, radius * 2, radius * 2)
        painter.setPen(QPen(QColor("#2b3243"), 6))
        painter.drawEllipse(ring.adjusted(3, 3, -3, -3))
        pen = QPen(QColor(BLUE if progress < 1 else ORANGE), 6)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawArc(ring.adjusted(3, 3, -3, -3), 90 * 16, int(-360 * 16 * progress))
        self._text(painter, ring, f"{max(0.0, 1 - progress):.1f}", 15, TEXT, bold=True)
        text_rect = QRectF(ring.right() + 12, box.top(), box.right() - ring.right() - 20,
                           box.height())
        font = QFont(self.font())
        font.setPixelSize(15)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor(TEXT))
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignVCenter, title)


def card(title: str, *widgets: QWidget) -> QFrame:
    frame = QFrame(objectName="card")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(16, 14, 16, 16)
    layout.setSpacing(10)
    layout.addWidget(QLabel(title.upper(), objectName="cardTitle"))
    for widget in widgets:
        layout.addWidget(widget)
    return frame


def toggle_row(text: str, toggle: Toggle, detail: QLabel | None = None) -> QWidget:
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    labels = QVBoxLayout()
    labels.setSpacing(2)
    labels.addWidget(QLabel(text))
    if detail is not None:
        labels.addWidget(detail)
    layout.addLayout(labels, 1)
    layout.addWidget(toggle, 0, Qt.AlignmentFlag.AlignVCenter)
    return row


class MainWindow(QMainWindow):
    def __init__(self, options: Options, start_worker: bool = True):
        super().__init__()
        self.setWindowTitle("HelCam")
        self.setWindowIcon(QIcon(str(ICON)))
        self.resize(1240, 760)
        self.options = options
        self.worker: VideoWorker | None = None

        self.view = VideoView()
        sidebar = self._build_sidebar()
        root = QWidget(objectName="root")
        layout = QHBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.view, 1)
        layout.addWidget(sidebar)
        self.setCentralWidget(root)

        QShortcut(QKeySequence(Qt.Key.Key_Space), self, self._recapture)
        QShortcut(QKeySequence(Qt.Key.Key_T), self, self._trigger)
        QShortcut(QKeySequence(Qt.Key.Key_F11), self, self._toggle_fullscreen)
        QShortcut(QKeySequence("Ctrl+Q"), self, self.close)

        if start_worker:
            self._start_worker()

    # Panel lateral ----------------------------------------------------------

    def _build_sidebar(self) -> QFrame:
        sidebar = QFrame(objectName="sidebar")
        sidebar.setFixedWidth(340)
        outer = QVBoxLayout(sidebar)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea(widgetResizable=True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QWidget()
        scroll.setWidget(content)
        outer.addWidget(scroll)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(20, 22, 20, 18)
        layout.setSpacing(14)

        header = QHBoxLayout()
        logo = QLabel()
        logo.setPixmap(QIcon(str(ICON)).pixmap(40, 40))
        titles = QVBoxLayout()
        titles.setSpacing(0)
        titles.addWidget(QLabel("HelCam", objectName="title"))
        titles.addWidget(QLabel("Teletransporte para tu webcam", objectName="subtitle"))
        header.addWidget(logo)
        header.addSpacing(6)
        header.addLayout(titles, 1)
        layout.addLayout(header)

        # Estado
        self.phase_label = QLabel("Iniciando…", objectName="phase")
        self.hand_chip = QLabel("Mano", objectName="chip")
        self.face_chip = QLabel("Cara", objectName="chip")
        chips = QWidget()
        chips_layout = QHBoxLayout(chips)
        chips_layout.setContentsMargins(0, 0, 0, 0)
        chips_layout.setSpacing(6)
        chips_layout.addWidget(self.hand_chip)
        chips_layout.addWidget(self.face_chip)
        chips_layout.addStretch()
        self.phase_hint = QLabel("", objectName="hint", wordWrap=True)
        # Las etiquetas con salto de línea no piden altura dentro del scroll: se reserva.
        self.phase_hint.setMinimumHeight(self.phase_hint.fontMetrics().lineSpacing() * 2 + 4)
        layout.addWidget(card("Estado", self.phase_label, chips, self.phase_hint))
        self._set_chip(self.hand_chip, False)
        self._set_chip(self.face_chip, False)

        # Acciones
        self.teleport_button = QPushButton("Teletransportarme", objectName="primary")
        self.teleport_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.teleport_button.clicked.connect(self._trigger)
        self.teleport_button.setEnabled(False)
        recapture = QPushButton("Recapturar fondo")
        recapture.setCursor(Qt.CursorShape.PointingHandCursor)
        recapture.clicked.connect(self._recapture)
        layout.addWidget(self.teleport_button)
        layout.addWidget(recapture)

        # Estilo
        segments = QFrame(objectName="segments")
        segment_layout = QHBoxLayout(segments)
        segment_layout.setContentsMargins(3, 3, 3, 3)
        segment_layout.setSpacing(3)
        self.style_group = QButtonGroup(self)
        for style, text in ((Style.ANIME, "Anime"), (Style.VFX, "VFX")):
            button = QPushButton(text, objectName="segment", checkable=True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setChecked(style == self.options.style)
            button.setProperty("style_value", style.value)
            self.style_group.addButton(button)
            segment_layout.addWidget(button)
        self.style_group.buttonClicked.connect(
            lambda button: self._send("style", button.property("style_value")))
        self.style_hint = QLabel(objectName="hint", wordWrap=True)
        self.style_group.buttonClicked.connect(lambda _: self._update_style_hint())
        layout.addWidget(card("Efecto", segments, self.style_hint))
        self._update_style_hint()

        # Cámara
        self.camera_box = QComboBox()
        for device in webcams():
            self.camera_box.addItem(device.label, device.index)
        index = self.camera_box.findData(self.options.camera)
        if index >= 0:
            self.camera_box.setCurrentIndex(index)
        self.camera_box.currentIndexChanged.connect(
            lambda _: self._send("camera", self.camera_box.currentData()))
        self.mirror_toggle = Toggle(self.options.mirror)
        self.mirror_toggle.toggled.connect(lambda on: self._send("mirror", on))
        layout.addWidget(card("Cámara", self.camera_box,
                              toggle_row("Vista previa en espejo", self.mirror_toggle)))

        # Salida y sonido
        self.output_detail = QLabel("Apagada", objectName="hint", wordWrap=True)
        self.output_detail.setMinimumHeight(self.output_detail.fontMetrics().lineSpacing() * 2 + 4)
        self.virtual_toggle = Toggle(self.options.virtual)
        self.virtual_toggle.toggled.connect(lambda on: self._send("virtual", on))
        self.sound_toggle = Toggle(self.options.sound)
        self.sound_toggle.toggled.connect(lambda on: self._send("sound", on))
        layout.addWidget(card("Salida", toggle_row("Cámara virtual", self.virtual_toggle,
                                                   self.output_detail),
                              toggle_row("Sonido", self.sound_toggle)))

        layout.addStretch()
        layout.addWidget(QLabel(
            "Índice y medio juntos en la frente, 1 segundo.\n"
            "Espacio: recapturar fondo · T: teletransportarme · F11: pantalla completa",
            objectName="hint", wordWrap=True))
        return sidebar

    def _set_chip(self, chip: QLabel, on: bool) -> None:
        chip.setStyleSheet(
            f"background: {'#183b2a' if on else '#222836'}; color: {GREEN if on else MUTED};"
            "border-radius: 10px; padding: 3px 10px; font-size: 12px; font-weight: 600;")

    def _update_style_hint(self) -> None:
        button = self.style_group.checkedButton()
        anime = button is None or button.property("style_value") == Style.ANIME.value
        self.style_hint.setText(
            "Parpadeo rápido que se rompe en tiras verticales, como en la serie." if anime
            else "Aura mientras sostienes el gesto, línea de luz y onda expansiva.")

    # Hilo de video ----------------------------------------------------------

    def _start_worker(self) -> None:
        self.worker = VideoWorker(self.options)
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.loading.connect(self.view.set_loading)
        self.worker.output_changed.connect(self._on_output)
        self.worker.problem.connect(self._on_problem)
        self.worker.start()

    def _send(self, command: str, value: object = None) -> None:
        if self.worker is not None:
            self.worker.send(command, value)
        settings = QSettings("helcam", "helcam")
        if command in ("style", "mirror", "virtual", "sound", "camera"):
            settings.setValue(command, value)

    def _recapture(self) -> None:
        self._send("recapture")

    def _trigger(self) -> None:
        self._send("trigger")

    def _toggle_fullscreen(self) -> None:
        self.showNormal() if self.isFullScreen() else self.showFullScreen()

    def _on_frame(self, image: QImage, status: Status) -> None:
        self.view.set_frame(image, status)
        self.show_status(status)

    def show_status(self, status: Status) -> None:
        label, color = PHASES[status.phase]
        self.phase_label.setText(label)
        self.phase_label.setStyleSheet(f"color: {color};")
        self._set_chip(self.hand_chip, status.hands > 0)
        self._set_chip(self.face_chip, status.face)
        hints = {
            Phase.WAITING: "Sal por completo del cuadro: el fondo se captura solo.",
            Phase.CAPTURING: "Un momento… no muevas la cámara.",
            Phase.READY: "Pon índice y medio juntos en la frente durante 1 segundo.",
            Phase.VANISHING: "¡Shun!",
            Phase.RETURNING: "¡Shun!",
            Phase.HIDDEN: "Nadie te ve. Repite el gesto para volver.",
        }
        self.phase_hint.setText(hints[status.phase])
        self.teleport_button.setEnabled(status.phase in (Phase.READY, Phase.HIDDEN))
        self.teleport_button.setText(
            "Regresar" if status.phase == Phase.HIDDEN else "Teletransportarme")

    def _on_output(self, device: str) -> None:
        self.view.set_live(device)
        self.output_detail.setText("Activa: en Meet elige «VanishCam»" if device else "Apagada")
        self.output_detail.setToolTip(device)
        if bool(device) != self.virtual_toggle.isChecked():
            self.virtual_toggle.set_quietly(bool(device))

    def _on_problem(self, text: str) -> None:
        QMessageBox.warning(self, "HelCam", text)

    def closeEvent(self, event) -> None:
        if self.worker is not None:
            self.worker.stop()
        super().closeEvent(event)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="helcam", description=__doc__)
    parser.add_argument("--camera", type=int, help="índice de la webcam (/dev/videoN)")
    parser.add_argument("--style", choices=[style.value for style in Style])
    parser.add_argument("--no-virtual", action="store_true", help="no mandar a la cámara virtual")
    return parser.parse_args(argv)


def load_options(args: argparse.Namespace) -> Options:
    settings = QSettings("helcam", "helcam")

    def saved(key: str, default):
        value = settings.value(key, default)
        if isinstance(default, bool):
            return value in (True, "true", "1", 1)
        return type(default)(value)

    cameras = webcams()
    camera = args.camera if args.camera is not None else saved(
        "camera", cameras[0].index if cameras else 0)
    return Options(
        camera=camera,
        style=Style(args.style or saved("style", Style.ANIME.value)),
        mirror=saved("mirror", True),
        virtual=False if args.no_virtual else saved("virtual", True),
        sound=saved("sound", True),
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    app = QApplication(sys.argv[:1])
    app.setApplicationName("HelCam")
    app.setDesktopFileName("helcam")
    app.setWindowIcon(QIcon(str(ICON)))
    app.setStyleSheet(STYLESHEET)
    window = MainWindow(load_options(args))
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
