import json

import pytest
from PIL import Image

from scripts.score import parse_output
from src import data_utils as du

W, H = 640, 360


def test_yolo_to_box_norm1000():
    b = du.yolo_to_box("pothole", [0.5, 0.5, 0.2, 0.2])
    assert b == {"label": "pothole", "bbox_2d": [400, 400, 600, 600]}


def test_yolo_to_box_pixel():
    b = du.yolo_to_box("crack", [0.5, 0.5, 0.5, 0.5], coord="pixel", width=W, height=H)
    assert b["bbox_2d"] == [160, 90, 480, 270]


def test_box_is_clamped():
    b = du.yolo_to_box("crack", [0.99, 0.01, 0.1, 0.1])["bbox_2d"]
    assert b[2] == 1000 and b[1] == 0


def test_tiny_box_is_never_degenerate():
    # parse_output loại cả output nếu x2 <= x1: nhãn không được sinh box suy biến
    for yolo in ([0.5, 0.5, 0.0001, 0.0001], [1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]):
        for coord in du.COORD_MODES:
            x1, y1, x2, y2 = du.yolo_to_box("crack", yolo, coord, W, H)["bbox_2d"]
            assert x2 > x1 and y2 > y1


def test_answer_sorted_top_to_bottom_left_to_right():
    objs = [{"label": "crack", "yolo": [0.8, 0.8, 0.1, 0.1]},
            {"label": "pothole", "yolo": [0.2, 0.2, 0.1, 0.1]},
            {"label": "manhole", "yolo": [0.6, 0.2, 0.1, 0.1]}]
    labels = [b["label"] for b in json.loads(du.objects_to_answer(objs))]
    assert labels == ["pothole", "manhole", "crack"]


def test_answer_is_compact_and_empty_is_list():
    assert du.objects_to_answer([]) == "[]"
    ans = du.objects_to_answer([{"label": "crack", "yolo": [0.5, 0.5, 0.2, 0.2]}])
    assert " " not in ans


@pytest.mark.parametrize("coord", du.COORD_MODES)
def test_roundtrip_to_pixels_matches_gt(coord):
    entry = {"width": W, "height": H, "objects": [{"label": "pothole", "yolo": [0.31, 0.62, 0.12, 0.2]}]}
    ans = du.objects_to_answer(entry["objects"], coord, W, H)
    pred = du.answer_to_pixels(ans, coord, W, H)[0]["bbox"]
    gt = du.gt_pixel_boxes(entry)[0]["bbox"]
    # làm tròn nguyên trên thang 0-1000 -> sai số tối đa 0.5 đơn vị = 0.32 px (ngang), 0.18 px (dọc)
    assert all(abs(a - b) <= 0.7 for a, b in zip(pred, gt))


def test_scorer_accepts_our_answer():
    ans = du.objects_to_answer([{"label": "crack", "yolo": [0.5, 0.5, 0.2, 0.2]}])
    assert parse_output(ans) == [{"label": "crack", "bbox": (400.0, 400.0, 600.0, 600.0)}]


def test_prompt_mentions_scale_and_classes():
    p = du.build_prompt("norm1000")
    assert "0-1000" in p and all(c in p for c in ("pothole", "crack", "manhole"))
    q = du.build_prompt("pixel", 640, 360)
    assert "640x360" in q and "0-1000" not in q
    with pytest.raises(ValueError):
        du.build_prompt("foo")


def _write_split(tmp_path, name, rows):
    d = tmp_path / "splits"
    d.mkdir(exist_ok=True)
    (d / f"{name}.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def _rows(n, session="s"):
    return [{"image_id": f"{session}{i}", "image_path": f"images/{i}.jpg", "width": W, "height": H,
             "session": f"{session}{i % 2}", "objects": [{"label": "crack", "yolo": [0.5, 0.5, 0.1, 0.1]}]}
            for i in range(n)]


def test_load_split_fraction_is_deterministic_and_ordered(tmp_path):
    _write_split(tmp_path, "train", _rows(100))
    a = du.load_split("train", tmp_path, fraction=0.1, seed=1)
    b = du.load_split("train", tmp_path, fraction=0.1, seed=1)
    assert len(a) == 10 and a == b
    ids = [int(e["image_id"][1:]) for e in a]
    assert ids == sorted(ids)
    assert len(du.load_split("train", tmp_path)) == 100


def test_load_split_limit_is_spread_out(tmp_path):
    _write_split(tmp_path, "val", _rows(100))
    got = du.load_split("val", tmp_path, limit=10)
    assert len(got) == 10 and got[0]["image_id"] == "s0" and got[-1]["image_id"] == "s90"


def test_split_stats_and_session_overlap(tmp_path):
    rows = _rows(4, "a")
    assert du.split_stats(rows) == {"images": 4, "boxes": {"pothole": 0, "crack": 4, "manhole": 0}, "sessions": 2}
    assert du.session_overlap({"train": rows, "val": _rows(2, "b")}) == {}
    assert set(du.session_overlap({"train": rows, "test": rows[:1]})) == {"a0"}


def test_render_prediction_marks_invalid_and_returns_image(tmp_path):
    (tmp_path / "images").mkdir()
    Image.new("RGB", (W, H), (120, 120, 120)).save(tmp_path / "images" / "0.jpg")
    entry = _rows(1)[0]
    ok = du.render_prediction(entry, du.objects_to_answer(entry["objects"]), tmp_path)
    bad = du.render_prediction(entry, "không phải json", tmp_path)
    assert ok.size == bad.size == (W, H + 16)
    assert ok.tobytes() != bad.tobytes()
    assert du.make_gallery([ok, bad, ok], cols=2).size == (2 * W, 2 * (H + 16))
