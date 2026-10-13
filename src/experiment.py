"""Một thí nghiệm hoàn chỉnh: nạp -> setup -> train -> NẠP LẠI từ adapter -> suy luận -> chấm (E1..E4)."""
import copy
import gc
import json
import shutil
from pathlib import Path

from src.data_utils import load_split
from src.inference import load_for_inference, run_inference, score_run
from src.model_utils import apply_setup_from_cfg, load_model
from src.training import train


def _restore_from_drive(run_dir, copy_to):
    """Session mới: work/ đã mất nhưng Drive còn adapter -> chép về work/<tag>."""
    src = Path(copy_to)
    if not (run_dir / "final").exists() and (src / "final").exists() and (src / "run.json").exists():
        shutil.copytree(src, run_dir, dirs_exist_ok=True)
        print(f"đã khôi phục adapter từ {src}")


def run_experiment(cfg, tag, work_dir="work", data_dir="data", evidence_dir="report/evidence",
                   copy_to=None, split="test", debug=False, reuse_adapter=False):
    """cfg: dict từ configs/eN.yaml (đã sửa đúng MỘT biến nếu là E4). tag: 'e1', 'e4_data_size_0.1', ...

    Suy luận luôn chạy trên model nạp lại từ adapter đã lưu (không dùng model vừa train), nhờ đó
    lỗi quên lưu/nạp projector lộ ra ngay thay vì làm E2/E3 trông giống E1.
    debug=True: 5 bước train, 8 ảnh val, bằng chứng ghi vào work/debug (không đụng report/evidence).
    reuse_adapter=True: nếu đã có adapter + run.json (ở work/<tag> hoặc Drive `copy_to`) thì BỎ QUA train,
    chỉ suy luận và chấm lại. Dùng khi session chết sau khi train xong. Bỏ qua khi debug=True.
    """
    import torch

    cfg = copy.deepcopy(cfg)
    coord = cfg.get("coord", "norm1000")
    if debug:
        split, evidence_dir, steps, limit = "val", Path(work_dir) / "debug", 5, 8
        tag, reuse_adapter = f"{tag}_debug", False  # debug có thư mục riêng, không đè adapter thật
    else:
        steps, limit = None, None
    run_dir = Path(work_dir) / tag

    if reuse_adapter and copy_to:
        _restore_from_drive(run_dir, copy_to)
    if reuse_adapter and (run_dir / "final" / "adapter_config.json").exists() and (run_dir / "run.json").exists():
        run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        print(f"BỎ QUA train, dùng adapter có sẵn ở {run_dir / 'final'}")
    else:
        model, processor = load_model()
        model, info = apply_setup_from_cfg(model, cfg)
        print("setup:", info)
        train_entries = load_split("train", data_dir, fraction=cfg.get("train_fraction", 1.0), seed=cfg.get("seed", 42))
        run = train(model, processor, train_entries, cfg, run_dir, data_dir, max_steps=steps, copy_to=copy_to)
        run["setup"] = info
        del model
        gc.collect()
        torch.cuda.empty_cache()
    print({k: run[k] for k in ("trainable_params", "peak_vram_gb", "train_minutes", "final_loss")})

    model, processor = load_for_inference(run_dir / "final")
    entries = load_split(split, data_dir, limit=limit)
    pred = run_inference(model, processor, entries, Path(work_dir) / f"{tag}.jsonl", data_dir, coord)
    result = score_run(pred, tag.upper(), Path(evidence_dir) / f"{tag}.json", run_dir / "run.json",
                       split=split, coord=coord, data_dir=data_dir)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result
