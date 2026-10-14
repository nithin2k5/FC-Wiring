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
@pytest.mark.parametrize("dx, dy", [(60, 0), (-80, 30), (40, -50)])
def test_sample_image_connectors_moved(dx, dy):
    result = inspect(shift(cv2.imread(str(SAMPLE)), dx, dy), CONFIG, "FCFK2FF3")
    assert result.passed, [r.message for r in result.regions]
    for region in result.regions:
        assert abs(region.offset[0] - dx) <= 2 and abs(region.offset[1] - dy) <= 2


def connector_frame(dx=0, dy=0):
    """Two wires hanging below a patterned connector body, moved by (dx, dy)."""
    frame = np.full((300, 300, 3), 255, np.uint8)
    x, y = 100 + dx, 120 + dy
    cv2.rectangle(frame, (x, y - 70), (x + 80, y), (40, 180, 40), -1)
    cv2.circle(frame, (x + 40, y - 35), 20, (20, 20, 20), -1)
    cv2.rectangle(frame, (x + 25, y), (x + 34, y + 90), BLACK, -1)
    cv2.rectangle(frame, (x + 46, y), (x + 55, y + 90), BLUE, -1)
    return frame


def anchored_config():
    box = [95, 125, 185, 205]
    region = {"name": "R", "box": box, "expected_order": ["black", "blue"],
              "anchor": make_anchor(connector_frame(), box, [300, 300])}
    return {"colors": CONFIG["colors"], "detection": CONFIG["detection"],
            "profiles": {"test": {"image_size": [300, 300], "regions": [region]}}}


def test_anchor_follows_moved_connector():
    config = anchored_config()
    assert inspect(connector_frame(), config, "test").passed
    result = inspect(connector_frame(dx=-60, dy=45), config, "test")
    assert result.passed, result.regions[0].message
    assert result.regions[0].offset == (-60, 45)


def test_anchor_without_connector_fails():
    result = inspect(np.full((300, 300, 3), 255, np.uint8), anchored_config(), "test")
    assert not result.passed
    assert "Connector not found" in result.regions[0].message
