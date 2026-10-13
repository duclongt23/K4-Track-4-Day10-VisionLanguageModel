"""Suy luận trên split và chấm bằng scripts/score.py (GUIDE bước 3 và 7)."""
import json
import subprocess
import sys
from pathlib import Path

from src.data_utils import build_prompt, load_image
from src.model_utils import EXTRA_FILE, MODEL_ID, load_extra, load_model

REPO_ROOT = Path(__file__).resolve().parents[1]
META_KEYS = ("trainable_params", "peak_vram_gb", "train_minutes")


def load_for_inference(adapter_dir=None, model_id=MODEL_ID):
    """Model gốc 4-bit -> (adapter LoRA) -> (projector đã huấn luyện, nếu có file extra_trainable.pt)."""
    model, processor = load_model(model_id)
    if adapter_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter_dir))
        extra = Path(adapter_dir) / EXTRA_FILE
        if extra.exists():
            print(f"nạp lại projector: {load_extra(model, extra)} tensor từ {extra}")
    model.eval()
    return model, processor


def run_inference(model, processor, entries, out_path, data_dir="data", coord="norm1000",
                  batch_size=4, max_new_tokens=256, resume=False):
    """Ghi mỗi dòng {"image_id", "raw"} vào out_path. resume=True: bỏ qua ảnh đã có (khi Colab bị ngắt)."""
    import torch

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if resume and out_path.exists():
        done = {json.loads(l)["image_id"] for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()}
    todo = [e for e in entries if e["image_id"] not in done]

    model.config.use_cache = True
    model.eval()
    with open(out_path, "a" if resume else "w", encoding="utf-8") as f:
        for i in range(0, len(todo), batch_size):
            chunk = todo[i:i + batch_size]
            messages = [[{"role": "user", "content": [
                {"type": "image", "image": load_image(e, data_dir)},
                {"type": "text", "text": build_prompt(coord, e["width"], e["height"])}]}] for e in chunk]
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_dict=True,
                return_tensors="pt", padding=True).to(model.device)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            raws = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            for e, raw in zip(chunk, raws):
                f.write(json.dumps({"image_id": e["image_id"], "raw": raw}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"\rsuy luận {min(i + batch_size, len(todo))}/{len(todo)}", end="")
    print()
    return out_path


def score_run(pred, name, out, run_json=None, split="test", coord="norm1000", data_dir="data"):
    """Gọi scripts/score.py (không sửa file đề). Có run_json thì gắn --meta trainable_params/peak_vram_gb/train_minutes."""
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "score.py"),
           "--pred", str(Path(pred).resolve()), "--name", name, "--out", str(Path(out).resolve()),
           "--split", split, "--coord", coord, "--splits-dir", str((Path(data_dir) / "splits").resolve())]
    if run_json:
        run = json.loads(Path(run_json).read_text(encoding="utf-8"))
        for k in META_KEYS:
            cmd += ["--meta", f"{k}={run[k]}"]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT, encoding="utf-8")
    print(res.stdout + res.stderr)
    res.check_returncode()
    return json.loads(Path(out).read_text(encoding="utf-8"))
