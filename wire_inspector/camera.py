"""Camera discovery and opening."""
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass

import cv2

# DirectShow opens far faster than the default MSMF backend on Windows.
BACKEND = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY

RESOLUTIONS = ["Default", "640x480", "1280x720", "1920x1080", "2560x1440", "3840x2160"]


@dataclass
class CameraInfo:
    index: int
    name: str
    width: int
    height: int

    @property
    def label(self) -> str:
        return f"{self.index}: {self.name}"


def _device_names() -> list[str]:
    """Best-effort camera names from Windows PnP (in enumeration order)."""
    if sys.platform != "win32":
        return []
    cmd = ("Get-PnpDevice -Class Camera,Image -PresentOnly -ErrorAction SilentlyContinue"
           " | Select-Object -ExpandProperty FriendlyName")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True,
                             text=True, timeout=5, creationflags=subprocess.CREATE_NO_WINDOW)
    except (OSError, subprocess.SubprocessError):
        return []
    return [line.strip() for line in out.stdout.splitlines() if line.strip()]


def list_cameras(max_index: int = 6) -> list[CameraInfo]:
    """Probe camera indices and return the ones that deliver frames."""
    found = []
    misses = 0
    for index in range(max_index):
        cap = cv2.VideoCapture(index, BACKEND)
        try:
            ok = cap.isOpened() and cap.read()[0]
            if ok:
                found.append(CameraInfo(index, f"Camera {index}",
                                        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
                misses = 0
            else:
                misses += 1
        finally:
            cap.release()
        if misses >= 2:  # indices are contiguous in practice; stop after a gap
            break

    # PnP order usually matches DirectShow order, but only trust it when the counts agree.
    names = _device_names()
    if len(names) == len(found):
        for cam, name in zip(found, names):
            cam.name = name
    return found


def open_camera(index: int, resolution: str = "Default"):
    cap = cv2.VideoCapture(index, BACKEND)
    if not cap.isOpened():
        cap.release()
        return None
    if resolution != "Default":
        w, h = (int(v) for v in resolution.split("x"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    return cap
