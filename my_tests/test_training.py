import pytest
import torch

from src import training as tr


class FakeTokenizer:
    pad_token_id = 0

    def convert_tokens_to_ids(self, tok):
        return {"<|im_end|>": 9, "<|image_pad|>": 5}[tok]

    def decode(self, ids):
        return " ".join(str(int(i)) for i in ids)


class FakeProcessor:
    """Mô phỏng chat template: prompt = [1,2,5,5], đầy đủ = prompt + [7,8,<|im_end|>,'\\n']."""
    tokenizer = FakeTokenizer()
    image_token_id = 5

    def __init__(self, prefix_ok=True):
        self.prefix_ok = prefix_ok

    def apply_chat_template(self, messages, add_generation_prompt, **kw):
        ids = [1, 2, 5, 5] if add_generation_prompt else [1, 2 if self.prefix_ok else 3, 5, 5, 7, 8, 9, 10]
        return {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
                "mm_token_type_ids": torch.zeros(1, len(ids), dtype=torch.long),
                "pixel_values": torch.zeros(4, 3), "image_grid_thw": torch.tensor([[1, 2, 2]])}


def test_encode_masks_prompt_and_trailing_newline():
    enc = tr.encode_example(FakeProcessor(), image=None, prompt="p", answer="a")
    assert enc["labels"].tolist() == [-100, -100, -100, -100, 7, 8, 9, -100]
    assert enc["input_ids"].tolist() == [1, 2, 5, 5, 7, 8, 9, 10]
    assert enc["input_ids"].dim() == 1 and enc["pixel_values"].shape == (4, 3)


def test_encode_rejects_non_prefix():
    with pytest.raises(ValueError):
        tr.encode_example(FakeProcessor(prefix_ok=False), None, "p", "a")


def test_check_example_counts_image_tokens_and_learned_text():
    enc = tr.encode_example(FakeProcessor(), None, "p", "a")
    n_img, learned = tr.check_example(FakeProcessor(), enc, verbose=False)
    assert n_img == 2 and learned == "7 8 9"


def test_collate_pads_sequences_and_concats_images():
    a = tr.encode_example(FakeProcessor(), None, "p", "a")
    b = {k: (v[:6] if k in tr.SEQ_KEYS else v) for k, v in a.items()}  # chuỗi ngắn hơn 2 token
    out = tr.collate([a, b], pad_id=0)
    assert out["input_ids"].shape == (2, 8)
    assert out["input_ids"][1, 6:].tolist() == [0, 0]
    assert out["labels"][1, 6:].tolist() == [-100, -100]
    assert out["attention_mask"][1, 6:].tolist() == [0, 0]
    assert out["mm_token_type_ids"][1, 6:].tolist() == [0, 0]
    assert out["pixel_values"].shape == (8, 3)          # cat, không pad
    assert out["image_grid_thw"].shape == (2, 3)


def test_lr_schedule_warmup_then_cosine_to_zero():
    total = 300
    f = [tr.lr_factor(s, total) for s in range(total)]
    assert f[0] == pytest.approx(1 / 15) and f[14] == pytest.approx(1.0)  # warmup 5% = 15 bước
    assert f[15] == pytest.approx(1.0)
    assert all(f[i] >= f[i + 1] for i in range(15, total - 1))
    assert f[-1] < 0.01 and all(v > 0 for v in f)
