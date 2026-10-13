"""Dữ liệu cho lab Day 10: đọc split, nhãn YOLO -> JSON, prompt, vẽ box.

Module thuần Python + Pillow, không cần GPU nên test được ở local.
Mọi hàm phụ thuộc thang toạ độ đều nhận `coord` ("norm1000" | "pixel") để E4 chỉ cần đổi một tham số.
"""
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

from scripts.labcommon import CLASSES, read_jsonl
from scripts.score import parse_output

COORD_MODES = ("norm1000", "pixel")
GT_COLOR, PRED_COLOR = (40, 200, 70), (230, 50, 50)


def _axis_scale(coord, width, height):
    """Giá trị lớn nhất của trục x, y trong thang toạ độ `coord`."""
    if coord == "norm1000":
        return 1000, 1000
    if coord == "pixel":
        return width, height
    raise ValueError(f"coord phải thuộc {COORD_MODES}, nhận {coord!r}")


# ---------------------------------------------------------------- đọc split
def load_split(name, data_dir="data", fraction=1.0, seed=42, limit=None):
    """Đọc data/splits/<name>.jsonl.

    fraction < 1: lấy ngẫu nhiên (cố định theo seed) một phần ảnh, giữ nguyên thứ tự gốc (E4 data_size).
    limit: lấy `limit` ảnh cách đều nhau để gỡ lỗi nhanh (không lấy liền nhau vì ảnh cùng phiên giống nhau).
    """
    entries = read_jsonl(Path(data_dir) / "splits" / f"{name}.jsonl")
    if fraction < 1.0:
        k = max(1, round(len(entries) * fraction))
        keep = set(random.Random(seed).sample(range(len(entries)), k))
        entries = [e for i, e in enumerate(entries) if i in keep]
    if limit and limit < len(entries):
        step = len(entries) / limit
        entries = [entries[int(i * step)] for i in range(limit)]
    return entries


def load_image(entry, data_dir="data"):
    return Image.open(Path(data_dir) / entry["image_path"]).convert("RGB")


def split_stats(entries):
    boxes = {c: 0 for c in CLASSES}
    for e in entries:
        for o in e["objects"]:
            boxes[o["label"]] += 1
    return {"images": len(entries), "boxes": boxes, "sessions": len({e["session"] for e in entries})}


def session_overlap(splits):
    """splits: {tên: entries}. Trả {session: [các split chứa nó]} cho session nằm ở >= 2 split (phải rỗng)."""
    seen = {}
    for name, entries in splits.items():
        for s in {e["session"] for e in entries}:
            seen.setdefault(s, []).append(name)
    return {s: v for s, v in seen.items() if len(v) > 1}


# ------------------------------------------------------- nhãn YOLO -> JSON
def _clamp_span(lo, hi, smax):
    """Làm tròn, kẹp [0, smax] và đảm bảo hi > lo (score.parse_output loại cả output nếu x2 <= x1)."""
    lo = min(max(int(round(lo)), 0), smax)
    hi = min(max(int(round(hi)), 0), smax)
    if hi <= lo:
        if lo >= smax:
            lo, hi = smax - 1, smax
        else:
            hi = lo + 1
    return lo, hi


def yolo_to_box(label, yolo, coord="norm1000", width=640, height=360):
    xc, yc, w, h = yolo
    sx, sy = _axis_scale(coord, width, height)
    x1, x2 = _clamp_span((xc - w / 2) * sx, (xc + w / 2) * sx, sx)
    y1, y2 = _clamp_span((yc - h / 2) * sy, (yc + h / 2) * sy, sy)
    return {"label": label, "bbox_2d": [x1, y1, x2, y2]}


def objects_to_boxes(objects, coord="norm1000", width=640, height=360):
    boxes = [yolo_to_box(o["label"], o["yolo"], coord, width, height) for o in objects]
    # trên -> dưới, trái -> phải: thứ tự ổn định để model học
    return sorted(boxes, key=lambda b: (b["bbox_2d"][1], b["bbox_2d"][0]))


def objects_to_answer(objects, coord="norm1000", width=640, height=360):
    return json.dumps(objects_to_boxes(objects, coord, width, height), separators=(",", ":"))


def build_prompt(coord="norm1000", width=640, height=360):
    if coord == "norm1000":
        scale = "Coordinates are integers on a 0-1000 scale relative to the image width and height (not pixels)."
    elif coord == "pixel":
        scale = f"Coordinates are integer pixels in the {width}x{height} image."
    else:
        raise ValueError(f"coord phải thuộc {COORD_MODES}, nhận {coord!r}")
    return (
        "Detect all road damage in this dashcam image. "
        f"The classes are: {', '.join(CLASSES)}. "
        "Answer with ONLY a JSON list and no other text. "
        'Each item has the form {"label": <class>, "bbox_2d": [x1, y1, x2, y2]}, '
        "where (x1, y1) is the top-left corner and (x2, y2) the bottom-right corner. "
        f"{scale} "
        "If there is no road damage, answer []."
    )


# ----------------------------------------------- quy về pixel và vẽ box
def to_pixel_boxes(boxes, coord, width, height):
    """boxes: [{"label", "bbox"|"bbox_2d"}] trong thang `coord` -> [{"label", "bbox"}] theo pixel."""
    sx, sy = _axis_scale(coord, width, height)
    out = []
    for b in boxes:
        x1, y1, x2, y2 = b.get("bbox") or b["bbox_2d"]
        out.append({"label": b["label"],
                    "bbox": (x1 / sx * width, y1 / sy * height, x2 / sx * width, y2 / sy * height)})
    return out


def gt_pixel_boxes(entry):
    """Box đúng (từ nhãn YOLO gốc, chưa làm tròn) theo pixel."""
    w, h = entry["width"], entry["height"]
    return [{"label": o["label"], "bbox": ((o["yolo"][0] - o["yolo"][2] / 2) * w, (o["yolo"][1] - o["yolo"][3] / 2) * h,
                                           (o["yolo"][0] + o["yolo"][2] / 2) * w, (o["yolo"][1] + o["yolo"][3] / 2) * h)}
            for o in entry["objects"]]


def answer_to_pixels(answer, coord="norm1000", width=640, height=360):
    """JSON model trả về -> box theo pixel. None nếu JSON hỏng."""
    parsed = parse_output(answer)
    return None if parsed is None else to_pixel_boxes(parsed, coord, width, height)


def draw_boxes(image, gt=None, pred=None):
    """Vẽ box (pixel) lên bản sao của ảnh: xanh = đúng, đỏ = dự đoán."""
    img = image.copy()
    d = ImageDraw.Draw(img)
    for boxes, color in ((gt or [], GT_COLOR), (pred or [], PRED_COLOR)):
        for b in boxes:
            x1, y1, x2, y2 = b["bbox"]
            d.rectangle([x1, y1, x2, y2], outline=color, width=2)
            d.text((x1 + 2, max(y1 - 11, 0)), b["label"], fill=color)
    return img


def render_prediction(entry, raw=None, data_dir="data", coord="norm1000", title=None):
    """Ảnh có box đúng (xanh) và box dự đoán (đỏ) kèm dải chú thích phía trên."""
    img = load_image(entry, data_dir)
    gt = gt_pixel_boxes(entry)
    pred, note = None, "no prediction"
    if raw is not None:
        pred = answer_to_pixels(raw, coord, entry["width"], entry["height"])
        note = "INVALID JSON" if pred is None else f"pred={len(pred)}"
    out = draw_boxes(img, gt, pred)
    band = Image.new("RGB", (out.width, 16), (30, 30, 30))
    ImageDraw.Draw(band).text((3, 2), f"{title or entry['image_id']} | gt={len(gt)} {note}", fill=(255, 255, 255))
    canvas = Image.new("RGB", (out.width, out.height + band.height))
    canvas.paste(band, (0, 0))
    canvas.paste(out, (0, band.height))
    return canvas


def make_gallery(images, cols=3):
    """Ghép nhiều ảnh cùng cỡ thành một lưới."""
    w, h = images[0].size
    rows = (len(images) + cols - 1) // cols
    canvas = Image.new("RGB", (w * cols, h * rows), (255, 255, 255))
    for i, im in enumerate(images):
        canvas.paste(im, ((i % cols) * w, (i // cols) * h))
    return canvas
