"""Một thí nghiệm hoàn chỉnh: nạp -> setup -> train -> NẠP LẠI từ adapter -> suy luận -> chấm (E1..E4)."""
import copy
import gc
from pathlib import Path

from src.data_utils import load_split
from src.inference import load_for_inference, run_inference, score_run
from src.model_utils import apply_setup_from_cfg, load_model
from src.training import train


def run_experiment(cfg, tag, work_dir="work", data_dir="data", evidence_dir="report/evidence",
                   copy_to=None, split="test", debug=False):
    """cfg: dict từ configs/eN.yaml (đã sửa đúng MỘT biến nếu là E4). tag: 'e1', 'e4_data_size_0.1', ...

    Suy luận luôn chạy trên model nạp lại từ adapter đã lưu (không dùng model vừa train), nhờ đó
    lỗi quên lưu/nạp projector lộ ra ngay thay vì làm E2/E3 trông giống E1.
    debug=True: 5 bước train, 8 ảnh val, bằng chứng ghi vào work/debug (không đụng report/evidence).
    """
    import torch

    cfg = copy.deepcopy(cfg)
    coord = cfg.get("coord", "norm1000")
    run_dir = Path(work_dir) / tag
    if debug:
        split, evidence_dir, steps, limit = "val", Path(work_dir) / "debug", 5, 8
        tag = f"{tag}_debug"
    else:
        steps, limit = None, None

    model, processor = load_model()
    model, info = apply_setup_from_cfg(model, cfg)
    print("setup:", info)
    train_entries = load_split("train", data_dir, fraction=cfg.get("train_fraction", 1.0), seed=cfg.get("seed", 42))
    run = train(model, processor, train_entries, cfg, run_dir, data_dir, max_steps=steps, copy_to=copy_to)
    run["setup"] = info
    print({k: run[k] for k in ("trainable_params", "peak_vram_gb", "train_minutes", "final_loss")})

    del model
    gc.collect()
    torch.cuda.empty_cache()

    model, processor = load_for_inference(run_dir / "final")
    entries = load_split(split, data_dir, limit=limit)
    pred = run_inference(model, processor, entries, Path(work_dir) / f"{tag}.jsonl", data_dir, coord)
    result = score_run(pred, tag.upper(), Path(evidence_dir) / f"{tag}.json", run_dir / "run.json",
                       split=split, coord=coord, data_dir=data_dir)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result
