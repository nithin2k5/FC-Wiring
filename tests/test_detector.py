from pathlib import Path

import cv2
import numpy as np
import pytest

from wire_engine import draw_result, inspect, load_config, locate_regions, make_anchor, wire_signature

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
YELLOW = (20, 210, 230)


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
    ([YELLOW, RED], ["yellow", "red"]),
    ([RED, YELLOW, BLACK, YELLOW], ["red", "yellow", "black", "yellow"]),
])
def test_three_and_four_wires_pass(wires, order):
    result = inspect(multi_wire_frame(wires), multi_wire_config(order), "test")
    assert result.passed, result.regions[0].message
    assert result.regions[0].detected_order == order


def test_red_and_yellow_swapped_fails():
    result = inspect(multi_wire_frame([RED, YELLOW]), multi_wire_config(["yellow", "red"]), "test")
    assert not result.passed
    assert result.regions[0].detected_order == ["red", "yellow"]


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
        box = [x - 5, 155, x + 85, 235]
        regions.append({"name": name, "box": box, "expected_order": order,
                        "anchor": make_anchor(ref, anchor_box, [600, 400]),
                        "wire_signature": wire_signature(ref, box, [600, 400], CONFIG["colors"], order,
                                                         CONFIG["detection"])})
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


def with_areas(config, areas):
    for region, area in zip(config["profiles"]["test"]["regions"], areas):
        region["search_box"] = area
    return config


def test_search_areas_decide_which_connector_is_which():
    # Area L is the top half, area R the bottom half. L's connector sits to the
    # RIGHT of R's, which the left-to-right rule alone would have swapped.
    config = with_areas(two_connector_config(), [[0, 0, 600, 200], [0, 200, 600, 400]])
    frame = np.full((400, 600, 3), 255, np.uint8)
    draw_connector(frame, 450, 100, BLACK, BLUE)   # L: black | blue
    draw_connector(frame, 50, 290, BLUE, BLACK)    # R: blue | black
    result = inspect(frame, config, "test")
    assert result.passed, [r.message for r in result.regions]
    assert [r.offset for r in result.regions] == [(350, -50), (-350, 140)]


def test_connector_outside_its_area_not_found():
    config = with_areas(two_connector_config(), [[0, 0, 300, 400], [300, 0, 600, 400]])
    frame = np.full((400, 600, 3), 255, np.uint8)
    draw_connector(frame, 100, 150, BLACK, BLUE)
    draw_connector(frame, 200, 150, BLUE, BLACK)   # R's connector, but in L's area
    result = inspect(frame, config, "test")
    assert result.regions[0].passed
    assert result.regions[1].message == "Connector and wires not found in its area"


def test_wires_box_stays_inside_area():
    # L's wires box spans y 155-235, but its area stops at y 220.
    config = with_areas(two_connector_config(), [[0, 0, 300, 220], [300, 0, 600, 400]])
    frame = np.full((400, 600, 3), 255, np.uint8)
    draw_connector(frame, 100, 150, BLACK, BLUE)
    draw_connector(frame, 400, 150, BLUE, BLACK)
    result = inspect(frame, config, "test")
    assert result.passed, [r.message for r in result.regions]
    assert result.regions[0].box == (95, 155, 185, 220)
    assert result.regions[0].search_box == (0, 0, 300, 220)


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample image missing")
@pytest.mark.parametrize("dx, dy", [(0, 0), (60, 40), (-80, -30)])
def test_sample_with_search_areas(dx, dy):
    import copy
    config = copy.deepcopy(CONFIG)
    left, right = config["profiles"]["FCFK2FF3"]["regions"]
    left["search_box"] = [150, 250, 560, 640]
    right["search_box"] = [660, 230, 1080, 640]
    result = inspect(shift(cv2.imread(str(SAMPLE)), dx, dy), config, "FCFK2FF3")
    assert result.passed, [r.message for r in result.regions]
    assert all(r.offset == (dx, dy) for r in result.regions)


# A black wire as the user's camera sees it (measured on a real frame): a blue
# cast, HSV H 116, S 89, V 55. Its saturation is inside blue's range, so the
# fixed ranges alone call it blue; only its low chroma (about 19) says black.
TINTED_BLACK = (55, 40, 36)
# A blue wire in deep shadow: dark enough for black's range, but clearly saturated.
SHADOWED_BLUE = (60, 25, 10)


def test_tinted_black_wire_is_black_not_blue():
    result = inspect(multi_wire_frame([TINTED_BLACK, BLUE]), multi_wire_config(["black", "blue"]), "test")
    assert result.passed, result.regions[0].message


def test_tinted_black_wire_does_not_count_as_blue():
    result = inspect(multi_wire_frame([TINTED_BLACK, (255, 255, 255)]), multi_wire_config(["blue"]), "test")
    assert not result.passed
    assert result.regions[0].message == "Not detected: blue"


def test_shadowed_blue_stays_blue():
    result = inspect(multi_wire_frame([TINTED_BLACK, SHADOWED_BLUE]), multi_wire_config(["black", "blue"]), "test")
    assert result.passed, result.regions[0].message


@pytest.mark.parametrize("gain", [0.6, 1.0, 1.5, 2.0])
def test_black_survives_exposure_changes(gain):
    frame = multi_wire_frame([TINTED_BLACK, BLUE]).astype(float) * gain
    result = inspect(np.clip(frame, 0, 255).astype(np.uint8), multi_wire_config(["black", "blue"]), "test")
    assert result.passed, result.regions[0].message


def test_result_image_flags_missing_connector_instead_of_stale_box():
    config = with_areas(two_connector_config(), [[0, 0, 300, 400], [300, 0, 600, 400]])
    frame = np.full((400, 600, 3), 255, np.uint8)
    draw_connector(frame, 400, 150, BLUE, BLACK)   # only R's connector is there
    result = inspect(frame, config, "test")
    out = draw_result(frame, result)
    # No box drawn at L's stored wires position (95, 155)-(185, 235)...
    assert (out[155, 95:186] == 255).all() and (out[155:236, 95] == 255).all()
    # ...but a red "not found" label at the top of L's area.
    assert (out[5:35, 0:300] != 255).any()


def draw_bootless(frame, x, y, left_bgr, right_bgr):
    """The connector without its black boot, looking too different to match its picture."""
    cv2.rectangle(frame, (x + 15, y - 30), (x + 65, y), (150, 150, 150), -1)
    cv2.rectangle(frame, (x + 25, y), (x + 34, y + 90), left_bgr, -1)
    cv2.rectangle(frame, (x + 46, y), (x + 55, y + 90), right_bgr, -1)


def bootless_frame(l_wires, r_wires, l_pos=(100, 150), r_pos=(400, 150)):
    frame = np.full((400, 600, 3), 255, np.uint8)
    draw_bootless(frame, *l_pos, *l_wires)
    draw_bootless(frame, *r_pos, *r_wires)
    return frame


AREAS = [[0, 0, 300, 400], [300, 0, 600, 400]]


def test_part_without_boot_is_found_by_its_wires():
    # The connector pictures were taken with the boot on; this part has none,
    # so the connector isn't recognised and the wires are found directly.
    config = with_areas(two_connector_config(), AREAS)
    frame = bootless_frame((BLACK, BLUE), (BLUE, BLACK))
    assert all(score == 0.0 for _, score in locate_regions(
        frame, config["profiles"]["test"]["regions"], [600, 400], 0.5))
    result = inspect(frame, config, "test")
    assert result.passed, [r.message for r in result.regions]
    assert [r.located_by for r in result.regions] == ["wires", "wires"]


def test_part_without_boot_moved_inside_its_area():
    config = with_areas(two_connector_config(), AREAS)
    frame = bootless_frame((BLACK, BLUE), (BLUE, BLACK), l_pos=(30, 260), r_pos=(470, 100))
    result = inspect(frame, config, "test")
    assert result.passed, [r.message for r in result.regions]
    assert result.regions[0].box[0] <= 30 + 25 and result.regions[0].box[2] >= 30 + 55


def test_part_without_boot_wrong_order_still_fails():
    config = with_areas(two_connector_config(), AREAS)
    result = inspect(bootless_frame((BLUE, BLACK), (BLUE, BLACK)), config, "test")
    assert not result.regions[0].passed
    assert result.regions[0].message == "Wrong order: got blue | black, expected black | blue"
    assert result.regions[1].passed


def test_part_without_boot_missing_wire_fails():
    config = with_areas(two_connector_config(), AREAS)
    result = inspect(bootless_frame((BLACK, (255, 255, 255)), (BLUE, BLACK)), config, "test")
    assert not result.regions[0].passed
    assert "blue" in result.regions[0].message


def test_without_area_no_wire_search():
    # Without a search area the wires aren't searched for, so a box can't wander
    # onto another connector's wires.
    result = inspect(bootless_frame((BLACK, BLUE), (BLUE, BLACK)), two_connector_config(), "test")
    assert all(r.message == "Connector not found" for r in result.regions)


def test_matching_connector_still_used_first():
    config = with_areas(two_connector_config(), AREAS)
    frame = np.full((400, 600, 3), 255, np.uint8)
    draw_connector(frame, 100, 150, BLACK, BLUE)
    draw_connector(frame, 400, 150, BLUE, BLACK)
    result = inspect(frame, config, "test")
    assert result.passed
    assert [r.located_by for r in result.regions] == ["connector", "connector"]


def test_foam_below_wires_does_not_pass_for_missing_black():
    # Black wire missing, dark-gray foam sleeve under the wires: the area search
    # must not slide the box onto the foam and call it the black wire.
    config = with_areas(two_connector_config(), AREAS)
    frame = bootless_frame(((255, 255, 255), BLUE), (BLUE, BLACK))
    cv2.rectangle(frame, (115, 240), (165, 330), (70, 70, 70), -1)   # foam under L's wires
    result = inspect(frame, config, "test")
    assert not result.regions[0].passed
    assert "black" in result.regions[0].message


def test_signature_recorded_from_setup_frame():
    region = two_connector_config()["profiles"]["test"]["regions"][0]
    assert region["wire_signature"] == {"widths": [10.0, 10.0], "centers": [34.5, 55.5]}


def test_fixture_in_area_does_not_pass_for_wires():
    # Real wires missing; the area also holds a blue bar and a dark block, like
    # the blue fixture and black plate above a real connector. They are blue
    # and black, but far too wide for this connector's wires.
    config = with_areas(two_connector_config(), AREAS)
    frame = bootless_frame(((255, 255, 255), (255, 255, 255)), (BLUE, BLACK))
    cv2.rectangle(frame, (20, 20), (75, 100), BLUE, -1)
    cv2.rectangle(frame, (78, 20), (110, 100), BLACK, -1)
    result = inspect(frame, config, "test")
    assert not result.regions[0].passed
    assert result.regions[1].passed


def test_no_wire_search_without_signature():
    config = with_areas(two_connector_config(), AREAS)
    for region in config["profiles"]["test"]["regions"]:
        del region["wire_signature"]
    result = inspect(bootless_frame((BLACK, BLUE), (BLUE, BLACK)), config, "test")
    assert all(r.message == "Connector and wires not found in its area" for r in result.regions)
