"""Reward IoU cho GRPO (E5, tuỳ chọn). Python thuần nên test được mà không cần GPU."""
from scripts.score import IOU_THR, iou_xyxy, parse_output

BASE_REWARD = 0.2  # thưởng cho JSON hợp lệ


def reward(completion, gt_boxes):
    """completion: chuỗi model sinh. gt_boxes: [{"label", "bbox"}] cùng thang với output (norm1000).

    JSON hỏng -> 0. Cả hai rỗng -> 1. Còn lại 0.2 + 0.8 * tổng IoU cặp khớp / max(n_gt, n_pred, 1);
    box bịa hoặc bỏ sót đều bị phạt qua mẫu số.
    """
    pred = parse_output(completion)
    if pred is None:
        return 0.0
    if not gt_boxes and not pred:
        return 1.0
    pairs = sorted(((iou_xyxy(g["bbox"], p["bbox"]), i, j)
                    for i, g in enumerate(gt_boxes) for j, p in enumerate(pred)
                    if g["label"] == p["label"]), reverse=True)
    used_g, used_p, total = set(), set(), 0.0
    for v, i, j in pairs:
        if v < IOU_THR:
            break
        if i in used_g or j in used_p:
            continue
        used_g.add(i)
        used_p.add(j)
        total += v
    return BASE_REWARD + (1 - BASE_REWARD) * total / max(len(gt_boxes), len(pred), 1)
