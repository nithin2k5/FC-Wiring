from pathlib import Path

import cv2
import numpy as np
import pytest

from wire_inspector.detector import inspect, load_config, make_anchor

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(ROOT / "config.json")
SAMPLE = ROOT / "samples" / "FCFK2FF3_sample.png"


def synthetic_frame(left_bgr, right_bgr):
    """White frame with two vertical wires inside a single region."""
    frame = np.full((200, 200, 3), 255, np.uint8)
    cv2.rectangle(frame, (80, 20), (89, 180), left_bgr, -1)
    cv2.rectangle(frame, (110, 20), (119, 180), right_bgr, -1)
    return frame


def synthetic_config():
    return {
        "colors": CONFIG["colors"],
        "detection": CONFIG["detection"],
        "profiles": {"test": {"regions": [
            {"name": "R", "box": [50, 10, 150, 190], "expected_order": ["black", "blue"]},
        ]}},
    }


BLACK = (20, 20, 20)
BLUE = (200, 80, 20)


def test_correct_order_passes():
    result = inspect(synthetic_frame(BLACK, BLUE), synthetic_config(), "test")
    assert result.passed
    assert result.regions[0].detected_order == ["black", "blue"]


def test_swapped_order_fails():
    result = inspect(synthetic_frame(BLUE, BLACK), synthetic_config(), "test")
    assert not result.passed
    assert "Wrong order" in result.regions[0].message


def test_missing_wire_fails():
    result = inspect(synthetic_frame(BLACK, (255, 255, 255)), synthetic_config(), "test")
    assert not result.passed
    assert "blue" in result.regions[0].message


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample image missing")
def test_sample_image():
    frame = cv2.imread(str(SAMPLE))
    assert inspect(frame, CONFIG, "FCFK2FF3").passed

    # Mirror each region so the wires swap sides.
    for region in inspect(frame, CONFIG, "FCFK2FF3").regions:
        x1, y1, x2, y2 = region.box
        frame[y1:y2, x1:x2] = frame[y1:y2, x1:x2][:, ::-1].copy()
    result = inspect(frame, CONFIG, "FCFK2FF3")
    assert all(r.detected_order == ["blue", "black"] for r in result.regions)


RED = (30, 30, 220)


def multi_wire_frame(wires_bgr):
    """White frame with evenly spaced vertical wires, left to right."""
    frame = np.full((200, 200, 3), 255, np.uint8)
    for i, bgr in enumerate(wires_bgr):
        x = 60 + 22 * i
        cv2.rectangle(frame, (x, 20), (x + 9, 180), bgr, -1)
    return frame


def multi_wire_config(order):
    config = synthetic_config()
    config["profiles"]["test"]["regions"][0]["expected_order"] = order
    return config


@pytest.mark.parametrize("wires, order", [
    ([BLACK, BLUE, RED], ["black", "blue", "red"]),
    ([BLACK, BLUE, BLACK], ["black", "blue", "black"]),
    ([BLUE, BLACK, BLACK, RED], ["blue", "black", "black", "red"]),
])
def test_three_and_four_wires_pass(wires, order):
    result = inspect(multi_wire_frame(wires), multi_wire_config(order), "test")
    assert result.passed, result.regions[0].message
    assert result.regions[0].detected_order == order


def test_repeated_color_wrong_order_fails():
    result = inspect(multi_wire_frame([BLACK, BLACK, BLUE]),
                     multi_wire_config(["black", "blue", "black"]), "test")
    assert not result.passed
    assert "Wrong order" in result.regions[0].message


def test_repeated_color_missing_wire_fails():
    result = inspect(multi_wire_frame([BLACK, BLUE, (255, 255, 255)]),
                     multi_wire_config(["black", "blue", "black"]), "test")
    assert not result.passed
    assert "black (1 of 2)" in result.regions[0].message


def shift(frame, dx, dy):
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(frame, m, (frame.shape[1], frame.shape[0]),
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample image missing")
@pytest.mark.parametrize("dx, dy", [(60, 0), (-250, 150), (100, -250)])
def test_sample_image_shifted(dx, dy):
    result = inspect(shift(cv2.imread(str(SAMPLE)), dx, dy), CONFIG, "FCFK2FF3")
    assert result.passed, [r.message for r in result.regions]
    assert all(r.offset == (dx, dy) for r in result.regions)


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample image missing")
@pytest.mark.parametrize("positions", [
    [(40, 30), (700, 420)],
    [(500, 520), (880, 40)],
    [(150, 300), (420, 310)],
])
def test_sample_connectors_anywhere(positions):
    """Cut each connector out of the sample and paste it elsewhere on a plain background."""
    sample = cv2.imread(str(SAMPLE))
    frame = cv2.GaussianBlur(
        np.random.default_rng(0).normal(150, 25, sample.shape).clip(0, 255).astype(np.uint8), (7, 7), 0)
    expected_offsets = []
    for region, (px, py) in zip(CONFIG["profiles"]["FCFK2FF3"]["regions"], positions):
        a, b = region["anchor"]["box"], region["box"]
        x1, y1 = min(a[0], b[0]) - 12, min(a[1], b[1]) - 12
        x2, y2 = max(a[2], b[2]) + 12, max(a[3], b[3]) + 12
        frame[py:py + y2 - y1, px:px + x2 - x1] = sample[y1:y2, x1:x2]
        expected_offsets.append((px - x1, py - y1))
    result = inspect(frame, CONFIG, "FCFK2FF3")
    assert result.passed, [r.message for r in result.regions]
    assert [r.offset for r in result.regions] == expected_offsets


def draw_connector(frame, x, y, left_bgr, right_bgr):
    """A patterned connector body at (x, y) with two wires hanging below it."""
    cv2.rectangle(frame, (x, y - 70), (x + 80, y), (40, 180, 40), -1)
    cv2.circle(frame, (x + 40, y - 35), 20, (20, 20, 20), -1)
    cv2.rectangle(frame, (x + 25, y), (x + 34, y + 90), left_bgr, -1)
    cv2.rectangle(frame, (x + 46, y), (x + 55, y + 90), right_bgr, -1)


def two_connector_config():
    """Identical connectors; the left one is wired black|blue, the right one blue|black."""
    ref = np.full((400, 600, 3), 255, np.uint8)
    draw_connector(ref, 100, 150, BLACK, BLUE)
    draw_connector(ref, 400, 150, BLUE, BLACK)
    regions = []
    for name, x, order in (("L", 100, ["black", "blue"]), ("R", 400, ["blue", "black"])):
        anchor_box = [x - 5, 75, x + 85, 155]
        regions.append({"name": name, "box": [x - 5, 155, x + 85, 235], "expected_order": order,
                        "anchor": make_anchor(ref, anchor_box, [600, 400])})
    return {"colors": CONFIG["colors"], "detection": CONFIG["detection"],
            "profiles": {"test": {"image_size": [600, 400], "regions": regions}}}


@pytest.mark.parametrize("left, right", [((100, 150), (400, 150)), ((20, 300), (480, 90)),
                                         ((250, 100), (330, 300))])
def test_identical_connectors_found_anywhere(left, right):
    frame = np.full((400, 600, 3), 255, np.uint8)
    draw_connector(frame, *left, BLACK, BLUE)
    draw_connector(frame, *right, BLUE, BLACK)
    result = inspect(frame, two_connector_config(), "test")
    assert result.passed, [r.message for r in result.regions]
    assert [r.offset for r in result.regions] == [(left[0] - 100, left[1] - 150),
                                                  (right[0] - 400, right[1] - 150)]


def test_one_connector_missing():
    frame = np.full((400, 600, 3), 255, np.uint8)
    draw_connector(frame, 300, 200, BLACK, BLUE)
    result = inspect(frame, two_connector_config(), "test")
    assert not result.passed
    assert ["Connector not found" in r.message for r in result.regions].count(True) == 1


def test_no_connector_fails():
    result = inspect(np.full((400, 600, 3), 255, np.uint8), two_connector_config(), "test")
    assert not result.passed
    assert all(r.message == "Connector not found" for r in result.regions)
