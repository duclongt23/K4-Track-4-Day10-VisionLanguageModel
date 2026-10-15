import pytest
import torch

from src import grpo
from src.rewards import entry_gt_boxes, reward


def test_advantages_are_centered_and_scaled():
    adv = grpo.group_advantages([0.2, 0.2, 1.0, 0.2])
    assert sum(adv) == pytest.approx(0.0, abs=1e-5)
    assert adv[2] > 0 > adv[0] and adv[0] == pytest.approx(adv[1])


def test_advantages_zero_when_group_is_identical():
    assert grpo.group_advantages([0.5, 0.5, 0.5, 0.5]) == [0.0, 0.0, 0.0, 0.0]


def test_completion_mask_stops_after_first_eos_inclusive():
    comp = torch.tensor([[7, 8, 9, 0, 0],     # 9 = <|im_end|>, phần sau là đệm
                         [7, 8, 7, 8, 7]])    # không có eos: bị cắt, giữ hết
    mask = grpo.completion_mask(comp, eos_ids=[9])
    assert mask.tolist() == [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]


def test_completion_mask_handles_several_eos_ids():
    comp = torch.tensor([[7, 5, 9, 9]])
    assert grpo.completion_mask(comp, eos_ids=[5, 9]).tolist() == [[1, 1, 0, 0]]


def test_slice_vision_takes_rows_of_the_right_images():
    grid = torch.tensor([[1, 2, 2]] * 4)       # mỗi ảnh 4 hàng
    pv = torch.arange(16 * 3).reshape(16, 3)
    sub_pv, sub_grid = grpo.slice_vision(pv, grid, 1, 3)
    assert sub_pv.shape == (8, 3) and torch.equal(sub_pv, pv[4:12]) and sub_grid.shape == (2, 3)


def test_slice_vision_rejects_mixed_sizes():
    with pytest.raises(ValueError):
        grpo.slice_vision(torch.zeros(5, 3), torch.tensor([[1, 2, 2], [1, 1, 1]]), 0, 1)


def test_reward_stats_counts_invalid_empty_and_boxes():
    texts = ['[{"label": "crack", "bbox_2d": [0, 0, 10, 10]}]', "[]", "hỏng"]
    st = grpo.reward_stats(texts, [1.0, 0.2, 0.0])
    assert st["reward"] == pytest.approx(1.2 / 3)
    assert st["invalid"] == pytest.approx(1 / 3) and st["empty"] == pytest.approx(1 / 3)
    assert st["boxes"] == pytest.approx(0.5)      # trung bình trên các câu hợp lệ: (1 + 0) / 2


def test_entry_gt_boxes_feeds_reward_perfectly():
    entry = {"objects": [{"label": "pothole", "yolo": [0.5, 0.5, 0.2, 0.2]}]}
    gt = entry_gt_boxes(entry)
    assert gt == [{"label": "pothole", "bbox": (400.0, 400.0, 600.0, 600.0)}]
    assert reward('[{"label": "pothole", "bbox_2d": [400, 400, 600, 600]}]', gt) == pytest.approx(1.0)
