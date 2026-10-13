import json

import pytest

from src.rewards import reward

GT = [{"label": "pothole", "bbox": (100.0, 100.0, 300.0, 300.0)}]


def out(*boxes):
    return json.dumps([{"label": l, "bbox_2d": list(b)} for l, b in boxes])


def test_perfect_prediction_gets_one():
    assert reward(out(("pothole", (100, 100, 300, 300))), GT) == pytest.approx(1.0)


def test_broken_json_gets_zero():
    assert reward("not json at all", GT) == 0.0
    assert reward('[{"label": "pothole", "bbox_2d": [1, 2, 3]}]', GT) == 0.0


def test_wrong_class_only_gets_base_reward():
    assert reward(out(("crack", (100, 100, 300, 300))), GT) == pytest.approx(0.2)


def test_extra_box_is_penalised_via_denominator():
    r = reward(out(("pothole", (100, 100, 300, 300)), ("crack", (500, 500, 600, 600))), GT)
    assert r == pytest.approx(0.2 + 0.8 * 1.0 / 2)


def test_empty_prediction_cannot_beat_a_real_detection():
    # chống reward hacking: trả [] khi có gt chỉ được thưởng JSON hợp lệ
    assert reward("[]", GT) == pytest.approx(0.2)
    assert reward(out(("pothole", (110, 110, 290, 290))), GT) > reward("[]", GT)


def test_both_empty_gets_one():
    assert reward("[]", []) == 1.0


def test_fabricated_box_when_no_gt():
    assert reward(out(("crack", (0, 0, 10, 10))), []) == pytest.approx(0.2)


def test_low_iou_does_not_match():
    assert reward(out(("pothole", (250, 250, 500, 500))), GT) == pytest.approx(0.2)


def test_partial_iou_scales_reward():
    r = reward(out(("pothole", (100, 100, 300, 250))), GT)  # IoU = 0.75
    assert r == pytest.approx(0.2 + 0.8 * 0.75)


def test_greedy_match_pairs_each_box_once():
    gt = [{"label": "crack", "bbox": (0.0, 0.0, 100.0, 100.0)}, {"label": "crack", "bbox": (0.0, 0.0, 100.0, 90.0)}]
    r = reward(out(("crack", (0, 0, 100, 100))), gt)
    assert r == pytest.approx(0.2 + 0.8 * 1.0 / 2)
