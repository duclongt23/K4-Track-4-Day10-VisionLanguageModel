"""Dữ liệu huấn luyện (chat template + mask loss), collate, lịch lr và vòng train (GUIDE bước 4 và 6)."""
import json
import math
import random
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from src.data_utils import build_prompt, load_image, objects_to_answer
from src.model_utils import EXTRA_FILE, count_trainable, save_extra

SEQ_KEYS = ("input_ids", "attention_mask", "mm_token_type_ids", "labels")  # tensor 1-D [L] sau encode
PAD_VALUE = {"attention_mask": 0, "mm_token_type_ids": 0, "labels": -100}


def load_config(path):
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ------------------------------------------------------ mã hoá một ví dụ
def encode_example(processor, image, prompt, answer):
    """Tokenize cả hội thoại; chỉ phần trả lời của assistant có nhãn, còn lại là -100."""
    user = {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}
    assistant = {"role": "assistant", "content": [{"type": "text", "text": answer}]}
    kw = dict(tokenize=True, return_dict=True, return_tensors="pt")
    prompt_enc = processor.apply_chat_template([user], add_generation_prompt=True, **kw)
    full_enc = processor.apply_chat_template([user, assistant], add_generation_prompt=False, **kw)

    n = prompt_enc["input_ids"].shape[1]
    if not torch.equal(full_enc["input_ids"][0, :n], prompt_enc["input_ids"][0]):
        raise ValueError("prompt không phải tiền tố của chuỗi đầy đủ: mask loss sẽ lệch")

    ids = full_enc["input_ids"][0]
    labels = ids.clone()
    labels[:n] = -100
    # template thêm "\n" sau <|im_end|>: giữ <|im_end|> làm token kết thúc, bỏ "\n" thừa
    im_end = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    ends = (ids == im_end).nonzero()
    if len(ends):
        labels[int(ends[-1]) + 1:] = -100

    enc = {k: (v[0] if k in SEQ_KEYS else v) for k, v in full_enc.items()}
    enc["labels"] = labels
    return enc


def check_example(processor, enc, verbose=True):
    """Kiểm tra bước 4: số token ảnh (640x360 phải là 220) và đúng là phần trả lời được học."""
    tok = processor.tokenizer
    image_id = getattr(processor, "image_token_id", None) or tok.convert_tokens_to_ids("<|image_pad|>")
    n_image = int((enc["input_ids"] == image_id).sum())
    learned = tok.decode(enc["input_ids"][enc["labels"] != -100])
    if verbose:
        print(f"độ dài chuỗi        : {enc['input_ids'].shape[0]}")
        print(f"token ảnh           : {n_image}")
        print(f"token có nhãn       : {int((enc['labels'] != -100).sum())}")
        print(f"giải mã phần có nhãn: {learned!r}")
    return n_image, learned


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, entries, processor, data_dir="data", coord="norm1000"):
        self.entries, self.processor, self.data_dir, self.coord = entries, processor, data_dir, coord

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        e = self.entries[i]
        prompt = build_prompt(self.coord, e["width"], e["height"])
        answer = objects_to_answer(e["objects"], self.coord, e["width"], e["height"])
        return encode_example(self.processor, load_image(e, self.data_dir), prompt, answer)


def collate(batch, pad_id):
    """Pad chuỗi (đệm phải) tới cùng độ dài; ghép pixel_values và image_grid_thw theo chiều 0, KHÔNG pad."""
    max_len = max(b["input_ids"].shape[0] for b in batch)
    out = {}
    for k in batch[0]:
        if k in SEQ_KEYS:
            fill = pad_id if k == "input_ids" else PAD_VALUE[k]
            out[k] = torch.stack([F.pad(b[k], (0, max_len - b[k].shape[0]), value=fill) for b in batch])
        else:
            out[k] = torch.cat([b[k] for b in batch], dim=0)
    return out


# --------------------------------------------------------- vòng huấn luyện
def lr_factor(step, total, warmup_frac=0.05):
    """Hệ số nhân lr tại bước `step` (đếm từ 0): warmup tuyến tính rồi cosine về 0."""
    warm = max(1, int(total * warmup_frac))
    if step < warm:
        return (step + 1) / warm
    progress = (step - warm) / max(1, total - warm)
    return 0.5 * (1 + math.cos(math.pi * progress))


def save_checkpoint(model, path, save_projector, copy_to=None):
    path = Path(path)
    model.save_pretrained(path)  # chỉ có adapter LoRA
    if save_projector:
        save_extra(model, path / EXTRA_FILE)
    if copy_to:  # Colab free hay bị ngắt: chép sang Drive ngay
        shutil.copytree(path, Path(copy_to) / path.name, dirs_exist_ok=True)


def _cycle(loader):
    while True:
        yield from loader


def train(model, processor, entries, cfg, out_dir, data_dir="data", max_steps=None, copy_to=None, log_every=10):
    """Huấn luyện `model` (đã apply_setup) và trả run_info {trainable_params, peak_vram_gb, train_minutes, ...}."""
    steps = max_steps or cfg["max_steps"]
    accum, bs, seed = cfg["grad_accum"], cfg["batch_size"], cfg.get("seed", 42)
    coord = cfg.get("coord", "norm1000")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    torch.manual_seed(seed)

    pad_id = processor.tokenizer.pad_token_id
    loader = torch.utils.data.DataLoader(
        TrainDataset(entries, processor, data_dir, coord), batch_size=bs, shuffle=True, num_workers=0,
        collate_fn=lambda b: collate(b, pad_id), generator=torch.Generator().manual_seed(seed))
    batches = _cycle(loader)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, steps))
    scaler = torch.amp.GradScaler("cuda")

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.train()
    device = "cuda"
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    history = []
    save_proj = bool(cfg["train_projector"])

    for step in range(1, steps + 1):
        step_loss = 0.0
        for _ in range(accum):
            batch = {k: v.to(device) for k, v in next(batches).items()}
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(**batch).loss / accum
            if not torch.isfinite(loss):
                raise FloatingPointError(f"loss={loss.item()} ở bước {step}: nghi tràn số fp16, "
                                         "giảm lr hoặc batch_size, và kiểm tra lại mask")
            scaler.scale(loss).backward()
            step_loss += loss.item()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()
        history.append(step_loss)
        if step % log_every == 0 or step == 1:
            print(f"step {step:4d}/{steps} loss={step_loss:.4f} lr={sched.get_last_lr()[0]:.2e} "
                  f"{(time.time() - t0) / 60:.1f} phút")
        if step % cfg.get("save_every", 100) == 0 and step != steps:
            save_checkpoint(model, out_dir / f"step{step}", save_proj, copy_to)

    run = {"name": cfg.get("name"), "steps": steps, "trainable_params": count_trainable(model),
           "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3),
           "train_minutes": round((time.time() - t0) / 60, 2), "final_loss": history[-1],
           "n_train_images": len(entries), "loss_history": history, "cfg": cfg}
    save_checkpoint(model, out_dir / "final", save_proj, copy_to)
    (out_dir / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    if copy_to:
        shutil.copy2(out_dir / "run.json", Path(copy_to) / "run.json")
    return run
