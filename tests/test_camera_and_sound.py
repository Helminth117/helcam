import wave

from helcam.camera import list_devices, virtual_device, webcams
from helcam.sound import GENERATED_NAME, teleport_sound_path, write_teleport_sound


def fake_node(sysfs, number, name, index=0, virtual=False):
    node = sysfs / f"video{number}"
    node.mkdir()
    (node / "name").write_text(name + "\n")
    (node / "index").write_text(f"{index}\n")
    target = sysfs / ("devices/virtual" if virtual else "devices/usb") / f"dev{number}"
    target.mkdir(parents=True)
    (node / "device").symlink_to(target)


def laptop(tmp_path):
    """Lo mismo que reporta la HP: normal, infrarroja, sus nodos de metadatos y VanishCam."""
    fake_node(tmp_path, 0, "HP HD Camera: HP HD Camera")
    fake_node(tmp_path, 1, "HP HD Camera: HP HD Camera", index=1)
    fake_node(tmp_path, 2, "HP HD Camera: HP IR Camera")
    fake_node(tmp_path, 3, "HP HD Camera: HP IR Camera", index=1)
    fake_node(tmp_path, 10, "VanishCam", virtual=True)
    return tmp_path


def test_lists_one_node_per_camera(tmp_path):
    devices = list_devices(laptop(tmp_path))
    assert [device.index for device in devices] == [0, 2, 10]


def test_webcams_put_infrared_last_and_skip_virtual(tmp_path):
    cameras = webcams(laptop(tmp_path))
    assert [(c.index, c.label) for c in cameras] == [
        (0, "HP HD Camera"), (2, "HP IR Camera (infrarroja)")]


def test_finds_vanishcam(tmp_path):
    device = virtual_device(laptop(tmp_path))
    assert device is not None and device.path == "/dev/video10"


def test_no_virtual_camera_loaded(tmp_path):
    fake_node(tmp_path, 0, "Webcam")
    assert virtual_device(tmp_path) is None


def test_custom_sound_wins_over_generated_one(tmp_path):
    config, cache = tmp_path / "config", tmp_path / "cache"
    assert teleport_sound_path(config, cache) == cache / GENERATED_NAME
    config.mkdir()
    custom = config / "Teleport.MP3"
    custom.write_bytes(b"ID3")
    assert teleport_sound_path(config, cache) == custom


def test_generated_sound_is_short_stereo_wav(tmp_path):
    path = tmp_path / "shun.wav"
    write_teleport_sound(path)
    with wave.open(str(path)) as sound:
        assert sound.getnchannels() == 2
        assert sound.getnframes() / sound.getframerate() < 0.6
