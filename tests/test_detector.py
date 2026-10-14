from pathlib import Path

import cv2
import numpy as np
import pytest

from wire_inspector.detector import inspect, load_config

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
