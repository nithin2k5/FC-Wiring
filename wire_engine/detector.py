"""Wire color detection and left-to-right order verification.

Each inspection region is cropped from the frame and converted to HSV. For
every expected color a mask is built, and the per-column fill ratio is used to
locate the wire: wires run vertically through the region, so they produce
narrow columns with a high fill ratio, while blobs such as the connector boot
or foam sleeve only cover part of the height.

Connectors can sit anywhere in the frame, so a region can carry an anchor: a
picture of the connector body, taken from the reference image. At inspection
every anchor is template-matched over the whole frame. Connectors of the same
type look alike, so the matches are assigned to regions in the regions'
configured left-to-right (or top-to-bottom) order, and each box moves with its
connector. A region can also have a search area (search_box): its connector is
then only looked for inside that area, which also decides which connector the
region belongs to, and its wires box is kept inside the area.

A connector doesn't always look like its picture (a part may come with or
without its rubber boot). So in a region with a search area, if the connector
isn't found or the wires box it gives doesn't hold the expected wires, the
wires box is also slid over the whole area to where the expected colors run
through it, and whichever placement finds more of the expected wires is used.
"""
from __future__ import annotations

import base64
import itertools
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
    offset: tuple[int, int] = (0, 0)  # shift found by the anchor, frame px
    anchor_score: float | None = None  # None if the region has no anchor
    search_box: tuple[int, int, int, int] | None = None  # search area, frame coordinates
    # How the wires box was placed: "connector" (followed its connector),
    # "wires" (found by sliding over the search area), "fixed" (configured
    # position), or None when nothing was found.
    located_by: str | None = "fixed"


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


def wire_masks(region_bgr: np.ndarray, colors: dict, names, min_sat: int = 110,
               min_chroma: int = 40) -> dict[str, np.ndarray]:
    """A 0/255 mask per color name for the pixels that look like that wire color."""
    blurred = cv2.GaussianBlur(region_bgr, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    kernel = np.ones((3, 3), np.uint8)

    # A pixel is "colored" if it is clearly saturated, or its chroma (brightest
    # minus darkest channel) is high. HSV saturation alone misleads on dark
    # pixels: a black wire with a slight blue cast can be as saturated as a pale
    # blue, while its chroma stays low. Chroma alone misleads the other way: a
    # blue wire in deep shadow has low chroma but stays highly saturated.
    # Achromatic colors (black, white) must not be colored; the rest must be.
    chroma = blurred.max(axis=2).astype(np.int16) - blurred.min(axis=2)
    colored = (hsv[..., 1] >= min_sat) | (chroma >= min_chroma)
    background = float(np.percentile(hsv[..., 2], 90))

    masks = {}
    for name in names:
        spec = colors[name]
        mask = color_mask(hsv, spec["hsv_ranges"]) > 0
        if "relative_dark" in spec:
            # Also count pixels much darker than the region's background, so a
            # black wire under bright light still counts as black.
            mask |= hsv[..., 2] <= spec["relative_dark"] * background
        mask &= ~colored if spec.get("achromatic") else colored
        masks[name] = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel)

    # Dark, saturated pixels (e.g. a blue wire in shadow) can fall inside the
    # black range; don't let a chromatic color's pixels count as black too.
    if "black" in masks:
        for name, mask in masks.items():
            if name != "black":
                masks["black"] = cv2.bitwise_and(masks["black"], cv2.bitwise_not(mask))
    return masks


def detect_wires(region_bgr: np.ndarray, colors: dict, color_counts: dict[str, int],
                 min_fill: float = 0.35, min_width: int = 3,
                 min_sat: int = 110, min_chroma: int = 40) -> list[WireDetection]:
    """Find up to color_counts[name] wires of each color, sorted left to right."""
    masks = wire_masks(region_bgr, colors, color_counts, min_sat, min_chroma)
    results = []
    for name, mask in masks.items():
        fill = (mask > 0).mean(axis=0)
        for start, end, score in _column_runs(fill, min_fill, min_width, color_counts[name]):
            cols = np.arange(start, end)
            weights = fill[start:end]
            center = float((cols * weights).sum() / weights.sum())
            results.append(WireDetection(name, True, center, start, end, score))
    return sorted(results, key=lambda d: d.center_x)


def find_wires_box(area_bgr: np.ndarray, colors: dict, names, box_w: int, box_h: int,
                   min_fill: float = 0.35, min_sat: int = 110,
                   min_chroma: int = 40) -> tuple[int, int, int, int] | None:
    """Where in area_bgr a box_w x box_h wires box best holds wires of all the colors.

    Every box position is scored by, for each color, the best fill of a column
    inside the box (how much of the box height that color covers). Positions
    that hold all the wires form a plateau of top scores; the box is put at
    its center so the wires sit in the middle. If no position holds every
    color, the one holding the most is used, so a missing wire is reported as
    missing. Returns (x1, y1, x2, y2) in area coordinates, or None if no wire
    of any of the colors is in the area.
    """
    ah, aw = area_bgr.shape[:2]
    box_w, box_h = min(box_w, aw), min(box_h, ah)
    if box_w < 3 or box_h < 3:
        return None
    masks = wire_masks(area_bgr, colors, names, min_sat, min_chroma)
    score = np.zeros((ah - box_h + 1, aw - box_w + 1), np.float32)
    weakest = np.full_like(score, np.inf)
    for mask in masks.values():
        cum = np.vstack([np.zeros((1, aw), np.float32), np.cumsum(mask > 0, axis=0, dtype=np.float32)])
        fill = (cum[box_h:] - cum[:-box_h]) / box_h               # column fill for each box top
        # Best column within each box: a sliding max over box_w columns.
        best = cv2.dilate(fill, np.ones((1, box_w), np.uint8), anchor=(0, 0))[:, :aw - box_w + 1]
        score += best
        weakest = np.minimum(weakest, best)
    complete = weakest >= min_fill
    if complete.any():
        score[~complete] = 0                                       # prefer boxes with every color
    top = float(score.max())
    if top < min_fill:
        return None
    plateau = (score >= top - 0.05 * len(masks)).astype(np.uint8)
    _, labels = cv2.connectedComponents(plateau)
    by, bx = np.unravel_index(int(np.argmax(score)), score.shape)
    ys, xs = np.nonzero(labels == labels[by, bx])
    x, y = int(round(xs.mean())), int(round(ys.mean()))
    return x, y, x + box_w, y + box_h


def scale_box(box, profile_size, frame_size):
    """Scale a box defined for profile_size (w, h) to frame_size (w, h)."""
    if not profile_size:
        return tuple(int(v) for v in box)
    sx = frame_size[0] / profile_size[0]
    sy = frame_size[1] / profile_size[1]
    x1, y1, x2, y2 = box
    return int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)


def make_anchor(frame_bgr: np.ndarray, anchor_box, image_size) -> dict | None:
    """Capture the connector picture inside anchor_box (image_size coordinates)."""
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = scale_box(anchor_box, image_size, (w, h))
    x1, x2 = sorted((max(0, x1), min(w, x2)))
    y1, y2 = sorted((max(0, y1), min(h, y2)))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    ok, png = cv2.imencode(".png", frame_bgr[y1:y2, x1:x2])
    if not ok:
        return None
    return {"box": [int(v) for v in anchor_box], "png": base64.b64encode(png.tobytes()).decode("ascii")}


_anchor_cache: dict[str, np.ndarray] = {}


def _anchor_image(anchor: dict) -> np.ndarray:
    key = anchor["png"]
    if key not in _anchor_cache:
        data = np.frombuffer(base64.b64decode(key), np.uint8)
        _anchor_cache[key] = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return _anchor_cache[key]


def find_anchor(frame_bgr: np.ndarray, template: np.ndarray, min_score: float,
                max_matches: int = 6) -> list[tuple[int, int, float]]:
    """Top-left (x, y, score) of the best non-overlapping matches, best first."""
    th, tw = template.shape[:2]
    if frame_bgr.shape[0] < th or frame_bgr.shape[1] < tw:
        return []
    # Search at half size for speed, then refine each hit at full size.
    small = cv2.resize(frame_bgr, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    small_t = cv2.resize(template, (max(1, tw // 2), max(1, th // 2)), interpolation=cv2.INTER_AREA)
    scores = cv2.matchTemplate(small, small_t, cv2.TM_CCOEFF_NORMED)
    sh, sw = small_t.shape[:2]
    matches = []
    for _ in range(max_matches):
        _, best, _, (bx, by) = cv2.minMaxLoc(scores)
        if best < min_score * 0.8:
            break
        scores[max(0, by - sh // 2):by + sh // 2 + 1, max(0, bx - sw // 2):bx + sw // 2 + 1] = -1
        x0, y0 = max(0, 2 * bx - 4), max(0, 2 * by - 4)
        window = frame_bgr[y0:min(frame_bgr.shape[0], 2 * by + th + 4),
                           x0:min(frame_bgr.shape[1], 2 * bx + tw + 4)]
        if window.shape[0] < th or window.shape[1] < tw:
            continue
        fine = cv2.matchTemplate(window, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, (fx, fy) = cv2.minMaxLoc(fine)
        if score >= min_score:
            matches.append((x0 + fx, y0 + fy, float(score)))
    return sorted(matches, key=lambda m: m[2], reverse=True)


def _consistent(placed, size) -> bool:
    """Matches must not overlap and must keep the regions' configured layout.

    Regions that both have a search area already know which connector is
    theirs, so their layout isn't checked against each other.
    """
    for (o1, m1, a1), (o2, m2, a2) in itertools.combinations(placed, 2):
        if abs(m1[0] - m2[0]) < size[0] / 2 and abs(m1[1] - m2[1]) < size[1] / 2:
            return False
        if a1 and a2:
            continue
        for axis in (0, 1):
            if abs(o1[axis] - o2[axis]) >= size[axis] / 2:
                if (o1[axis] < o2[axis]) != (m1[axis] < m2[axis]):
                    return False
                break
    return True


def search_area(region: dict, image_size, frame_size) -> tuple[int, int, int, int] | None:
    """The region's search area in frame coordinates, clipped to the frame, or None."""
    if not region.get("search_box"):
        return None
    w, h = frame_size
    x1, y1, x2, y2 = scale_box(region["search_box"], image_size, frame_size)
    x1, x2 = sorted((max(0, x1), min(w, x2)))
    y1, y2 = sorted((max(0, y1), min(h, y2)))
    return (x1, y1, x2, y2) if x2 - x1 >= 2 and y2 - y1 >= 2 else None


def locate_regions(frame_bgr: np.ndarray, regions: list[dict], image_size,
                   min_score: float) -> list[tuple[tuple[int, int], float] | None]:
    """For each anchored region, the (dx, dy) its connector moved and the match score.

    Returns None for regions without an anchor and ((0, 0), 0.0) for anchored
    regions whose connector wasn't found.
    """
    h, w = frame_bgr.shape[:2]
    anchored, origins, candidates, has_area = [], [], [], []
    for i, region in enumerate(regions):
        anchor = region.get("anchor")
        if not anchor:
            continue
        ax1, ay1, ax2, ay2 = scale_box(anchor["box"], image_size, (w, h))
        template = _anchor_image(anchor)
        if template.shape[:2] != (ay2 - ay1, ax2 - ax1):
            template = cv2.resize(template, (ax2 - ax1, ay2 - ay1), interpolation=cv2.INTER_AREA)
        anchored.append(i)
        origins.append((ax1, ay1, ax2 - ax1, ay2 - ay1))
        area = search_area(region, image_size, (w, h))
        if area:
            sx1, sy1, sx2, sy2 = area
            matches = [(sx1 + x, sy1 + y, score)
                       for x, y, score in find_anchor(frame_bgr[sy1:sy2, sx1:sx2], template, min_score)]
        else:
            matches = find_anchor(frame_bgr, template, min_score)
        candidates.append(matches + [None])
        has_area.append(area is not None)

    # Pick one match per region (or none), preferring more regions found, then
    # the higher total score, among layouts consistent with the configuration.
    best, best_key = None, None
    for combo in itertools.product(*candidates):
        placed = [((o[0], o[1]), (m[0], m[1]), a) for o, m, a in zip(origins, combo, has_area) if m]
        size = (min(o[2] for o in origins), min(o[3] for o in origins)) if origins else (0, 0)
        if not _consistent(placed, size):
            continue
        key = (len(placed), sum(m[2] for m in combo if m))
        if best_key is None or key > best_key:
            best, best_key = combo, key

    out: list = [None] * len(regions)
    for i, o, m in zip(anchored, origins, best or [None] * len(anchored)):
        out[i] = ((m[0] - o[0], m[1] - o[1]), m[2]) if m else ((0, 0), 0.0)
    return out


def inspect(frame_bgr: np.ndarray, config: dict, profile_name: str) -> InspectionResult:
    profile = config["profiles"][profile_name]
    colors = config["colors"]
    det_cfg = config.get("detection", {})
    min_fill = det_cfg.get("min_column_fill", 0.35)
    min_width = det_cfg.get("min_wire_width_px", 3)
    min_anchor = det_cfg.get("anchor_min_score", 0.5)
    min_sat = det_cfg.get("colored_min_saturation", 110)
    min_chroma = det_cfg.get("colored_min_chroma", 40)
    # Stricter fill for wires found by searching the area: a real wire runs
    # through the whole box, while foam or a boot next to a wire end only
    # fills part of it, so this keeps them from passing for a missing wire.
    search_fill = max(min_fill, det_cfg.get("wire_search_min_fill", 0.6))

    h, w = frame_bgr.shape[:2]
    located = locate_regions(frame_bgr, profile["regions"], profile.get("image_size"), min_anchor)
    regions = []
    for region, location in zip(profile["regions"], located):
        x1, y1, x2, y2 = scale_box(region["box"], profile.get("image_size"), (w, h))
        box_w, box_h = x2 - x1, y2 - y1
        expected = list(region["expected_order"])
        counts = Counter(expected)
        offset, anchor_score = location if location else ((0, 0), None)
        area = search_area(region, profile.get("image_size"), (w, h))
        bx1, by1, bx2, by2 = area or (0, 0, w, h)

        def check(box, fill=min_fill):
            """Clip box to the area and detect the wires in it: (box, detections, wires found)."""
            cx1, cx2 = sorted((max(bx1, box[0]), min(bx2, box[2])))
            cy1, cy2 = sorted((max(by1, box[1]), min(by2, box[3])))
            if cx2 - cx1 < 2 or cy2 - cy1 < 2:
                return (cx1, cy1, cx2, cy2), [], -1
            dets = detect_wires(frame_bgr[cy1:cy2, cx1:cx2], colors, counts, fill, min_width,
                                min_sat, min_chroma)
            found = Counter(d.color for d in dets)
            return (cx1, cy1, cx2, cy2), dets, sum(min(found[c], n) for c, n in counts.items())

        placed = None  # (box, detections, wires found, located_by)
        if anchor_score is None:
            placed = (*check((x1, y1, x2, y2)), "fixed")
        elif anchor_score >= min_anchor:
            placed = (*check((x1 + offset[0], y1 + offset[1], x2 + offset[0], y2 + offset[1])), "connector")
        # In a search area, also look for the wires themselves when the
        # connector wasn't found or its box doesn't hold every expected wire.
        if area and (placed is None or placed[2] < len(expected)):
            found_box = find_wires_box(frame_bgr[by1:by2, bx1:bx2], colors, counts, box_w, box_h,
                                       search_fill, min_sat, min_chroma)
            if found_box:
                fx1, fy1, fx2, fy2 = found_box
                alt = (*check((bx1 + fx1, by1 + fy1, bx1 + fx2, by1 + fy2), search_fill), "wires")
                if placed is None or alt[2] > placed[2]:
                    placed = alt

        if placed is None:
            result = RegionResult(region["name"], (x1, y1, x2, y2), expected, [], offset=offset,
                                  anchor_score=anchor_score, search_box=area, located_by=None)
            result.message = ("Connector and wires not found in its area" if area
                              else "Connector not found")
            regions.append(result)
            continue

        box, detections, _, located_by = placed
        result = RegionResult(region["name"], box, expected, [], offset=offset,
                              anchor_score=anchor_score, search_box=area, located_by=located_by)
        if box[2] - box[0] < 2 or box[3] - box[1] < 2:
            result.message = "Wires box is outside the " + ("search area" if area else "image")
            regions.append(result)
            continue

        result.detections = detections
        result.detected_order = [d.color for d in detections]
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
    unplaced = 0  # "not found" labels without an area, stacked in the top-left corner
    for region in result.regions:
        if region.search_box:
            sx1, sy1, sx2, sy2 = region.search_box
            cv2.rectangle(out, (sx1, sy1), (sx2, sy2), (255, 200, 0), 1)
        if region.located_by is None:
            # The connector wasn't found, so the wires box would only mark a
            # stale spot. Say so on the search area (or the frame) instead.
            if region.search_box:
                pos = (region.search_box[0] + 4, region.search_box[1] + 22)
            else:
                pos = (10, 28 + 26 * unplaced)
                unplaced += 1
            label = f"{region.name}: CONNECTOR NOT FOUND"
            # Dark outline so the label reads on any background.
            cv2.putText(out, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(out, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
            continue
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
