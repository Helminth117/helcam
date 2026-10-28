#!/usr/bin/env bash
# Agrega "HelCam" al menú de aplicaciones apuntando a este repo.
set -euo pipefail
repo="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
target="${XDG_DATA_HOME:-$HOME/.local/share}/applications/helcam.desktop"
mkdir -p "$(dirname "$target")"
cat > "$target" <<DESKTOP
[Desktop Entry]
Type=Application
Name=HelCam
Comment=Teletransporte estilo anime para tu webcam (Meet, Zoom, Discord)
Exec=$repo/.venv/bin/helcam
Icon=$repo/helcam/assets/icon.svg
Terminal=false
Categories=Video;AudioVideo;
StartupWMClass=helcam
DESKTOP
echo "Listo: $target"
