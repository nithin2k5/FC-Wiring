"""Wire color detection and left-to-right order verification.

Each inspection region is cropped from the frame and converted to HSV. For
every expected color a mask is built, and the per-column fill ratio is used to
locate the wire: wires run vertically through the region, so they produce
narrow columns with a high fill ratio, while blobs such as the connector boot
or foam sleeve only cover part of the height.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class WireDetection:
    color: str
    found: bool
    center_x: float = 0.0  # in region coordinates
    x_start: int = 0
    x_end: int = 0
    score: float = 0.0  # mean column fill of the detected run (0..1)


@dataclass
class RegionResult:
    name: str
    box: tuple[int, int, int, int]  # x1, y1, x2, y2 in frame coordinates
    expected_order: list[str]
    detected_order: list[str]
    detections: list[WireDetection] = field(default_factory=list)  # left to right
    passed: bool = False
    message: str = ""


@dataclass
class InspectionResult:
    profile: str
    regions: list[RegionResult]

    @property
    def passed(self) -> bool:
        return bool(self.regions) and all(r.passed for r in self.regions)


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config(config: dict, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def color_mask(hsv: np.ndarray, ranges: list) -> np.ndarray:
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in ranges:
        mask |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    return mask


def _column_runs(fill: np.ndarray, min_fill: float, min_width: int, count: int):
    """Return up to `count` (start, end, score) wire runs, strongest first.

    Columns above min_fill form runs. A run can merge the wire with an adjacent
    partial blob (connector boot, shadow), or two neighbouring wires of the same
    color, so each run is split into pieces around its fill peaks: the columns
    within 70% of a peak form one piece. Pieces are ranked by mean fill, since
    a wire covers most of the region height while blobs are wide but partial.
    """
    above = fill >= min_fill
    pieces = []
    x = 0
    n = len(fill)
    while x < n:
        if not above[x]:
            x += 1
            continue
        start = x
        while x < n and above[x]:
            x += 1
        end = x
        taken = np.zeros(end - start, bool)
        while not taken.all():
            sub = np.where(taken, -1.0, fill[start:end])
            peak = start + int(np.argmax(sub))
            if fill[peak] < min_fill:
                break
            floor = 0.7 * fill[peak]
            lo, hi = peak, peak + 1
            while lo > start and not taken[lo - 1 - start] and fill[lo - 1] >= floor:
                lo -= 1
            while hi < end and not taken[hi - start] and fill[hi] >= floor:
                hi += 1
            taken[lo - start:hi - start] = True
            if hi - lo >= min_width:
                pieces.append((lo, hi, float(fill[lo:hi].mean())))
    pieces.sort(key=lambda p: p[2], reverse=True)
    return pieces[:count]


def detect_wires(region_bgr: np.ndarray, colors: dict, color_counts: dict[str, int],
                 min_fill: float = 0.35, min_width: int = 3) -> list[WireDetection]:
    """Find up to color_counts[name] wires of each color, sorted left to right."""
    blurred = cv2.GaussianBlur(region_bgr, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    kernel = np.ones((3, 3), np.uint8)

    masks = {}
    for name in color_counts:
        mask = color_mask(hsv, colors[name]["hsv_ranges"])
        masks[name] = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Dark, saturated pixels (e.g. a blue wire in shadow) can fall inside the
    # black range; don't let a chromatic color's pixels count as black too.
    if "black" in masks:
        for name, mask in masks.items():
            if name != "black":
                masks["black"] = cv2.bitwise_and(masks["black"], cv2.bitwise_not(mask))

    results = []
    for name, mask in masks.items():
        fill = (mask > 0).mean(axis=0)
        for start, end, score in _column_runs(fill, min_fill, min_width, color_counts[name]):
            cols = np.arange(start, end)
            weights = fill[start:end]
            center = float((cols * weights).sum() / weights.sum())
            results.append(WireDetection(name, True, center, start, end, score))
    return sorted(results, key=lambda d: d.center_x)


def scale_box(box, profile_size, frame_size):
    """Scale a box defined for profile_size (w, h) to frame_size (w, h)."""
    if not profile_size:
        return tuple(int(v) for v in box)
    sx = frame_size[0] / profile_size[0]
    sy = frame_size[1] / profile_size[1]
    x1, y1, x2, y2 = box
    return int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)


def inspect(frame_bgr: np.ndarray, config: dict, profile_name: str) -> InspectionResult:
    profile = config["profiles"][profile_name]
    colors = config["colors"]
    det_cfg = config.get("detection", {})
    min_fill = det_cfg.get("min_column_fill", 0.35)
    min_width = det_cfg.get("min_wire_width_px", 3)

    h, w = frame_bgr.shape[:2]
    regions = []
    for region in profile["regions"]:
        x1, y1, x2, y2 = scale_box(region["box"], profile.get("image_size"), (w, h))
        x1, x2 = sorted((max(0, x1), min(w, x2)))
        y1, y2 = sorted((max(0, y1), min(h, y2)))
        expected = list(region["expected_order"])
        result = RegionResult(region["name"], (x1, y1, x2, y2), expected, [])

        if x2 - x1 < 2 or y2 - y1 < 2:
            result.message = "Region is outside the image"
            regions.append(result)
            continue

        crop = frame_bgr[y1:y2, x1:x2]
        counts = Counter(expected)
        result.detections = detect_wires(crop, colors, counts, min_fill, min_width)
        result.detected_order = [d.color for d in result.detections]
        found = Counter(result.detected_order)
        missing = []
        for c, n in counts.items():
            if found[c] < n:
                missing.append(c if n == 1 else f"{c} ({found[c]} of {n})")

        if missing:
            result.message = "Not detected: " + ", ".join(missing)
        elif result.detected_order == expected:
            result.passed = True
            result.message = "OK: " + " | ".join(expected)
        else:
            result.message = ("Wrong order: got " + " | ".join(result.detected_order)
                              + ", expected " + " | ".join(expected))
        regions.append(result)
    return InspectionResult(profile_name, regions)


def draw_result(frame_bgr: np.ndarray, result: InspectionResult) -> np.ndarray:
    out = frame_bgr.copy()
    for region in result.regions:
        x1, y1, x2, y2 = region.box
        color = (0, 200, 0) if region.passed else (0, 0, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{region.name}: {'PASS' if region.passed else 'FAIL'}"
        cv2.putText(out, label, (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color, 2, cv2.LINE_AA)
        for i, det in enumerate(region.detections):
            cv2.rectangle(out, (x1 + det.x_start, y1 + 2), (x1 + det.x_end - 1, y2 - 2),
                          (0, 255, 255), 1)
            cv2.putText(out, f"{i + 1}:{det.color}", (x1 + 2, y2 + 16 + 16 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    return out
