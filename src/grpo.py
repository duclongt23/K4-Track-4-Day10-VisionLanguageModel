"""GRPO với reward IoU (E5, tuỳ chọn). Vòng lặp tự viết, on-policy, beta = 0 (không KL).

Ý tưởng một bước:
  1. Với mỗi ảnh, cho model SINH n câu trả lời (lấy mẫu ngẫu nhiên).
  2. Chấm mỗi câu bằng reward IoU. Advantage = (reward - trung bình nhóm) / độ lệch chuẩn nhóm.
  3. Tăng xác suất của câu có advantage dương, giảm câu có advantage âm: loss = -advantage * logprob.
Vì cập nhật ngay trên mẫu vừa sinh (on-policy, một lần) nên không cần tỉ số PPO hay clipping.
"""
import json
import random
import time
from pathlib import Path

import torch

from src.data_utils import build_prompt, load_image
from src.model_utils import count_trainable
from src.rewards import entry_gt_boxes, reward
from src.training import save_checkpoint
from scripts.score import parse_output


# ------------------------------------------------ hàm thuần (test được ở local)
def group_advantages(rewards, eps=1e-4):
    """Chuẩn hoá reward trong một nhóm: cả nhóm bằng nhau thì advantage = 0 (không có gì để học)."""
    r = torch.tensor(rewards, dtype=torch.float32)
    return ((r - r.mean()) / (r.std(unbiased=False) + eps)).tolist()


def completion_mask(comp_ids, eos_ids):
    """1 cho các token tới hết token kết thúc đầu tiên (tính cả nó), 0 cho phần đệm phía sau.
    Câu không có token kết thúc (bị cắt) thì giữ nguyên toàn bộ."""
    n, length = comp_ids.shape
    is_eos = torch.zeros_like(comp_ids, dtype=torch.bool)
    for e in eos_ids:
        is_eos |= comp_ids == e
    first = torch.where(is_eos.any(1), is_eos.float().argmax(1), torch.full((n,), length - 1, device=comp_ids.device))
    return (torch.arange(length, device=comp_ids.device)[None] <= first[:, None]).long()


def slice_vision(pixel_values, grid, a, b):
    """Lấy phần ảnh của các chuỗi a..b. Cả nhóm dùng cùng một ảnh nên mỗi ảnh có đúng P hàng pixel_values."""
    if not bool((grid == grid[0]).all()):
        raise ValueError("các ảnh trong nhóm phải cùng kích thước")
    p = int(grid[0].prod())
    return pixel_values[a * p:b * p], grid[a:b]


def reward_stats(texts, rewards):
    """Số liệu theo dõi reward hacking của một nhóm câu trả lời."""
    parsed = [parse_output(t) for t in texts]
    valid = [p for p in parsed if p is not None]
    return {"reward": sum(rewards) / len(rewards),
            "invalid": sum(p is None for p in parsed) / len(parsed),
            "empty": sum(p == [] for p in parsed) / len(parsed),
            "boxes": sum(len(p) for p in valid) / len(valid) if valid else 0.0}


# ----------------------------------------------------------- sinh và tính logprob
def _eos_ids(model, processor):
    eos = model.generation_config.eos_token_id
    ids = set(eos if isinstance(eos, (list, tuple)) else [eos]) if eos is not None else set()
    ids.add(processor.tokenizer.convert_tokens_to_ids("<|im_end|>"))
    return sorted(ids)


def sample_group(model, processor, entry, data_dir, coord, n, max_new_tokens, temperature):
    """Sinh n câu trả lời cho cùng một ảnh. Trả (inputs, comp_ids, texts)."""
    image = load_image(entry, data_dir)
    prompt = build_prompt(coord, entry["width"], entry["height"])
    msgs = [[{"role": "user", "content": [{"type": "image", "image": image},
                                          {"type": "text", "text": prompt}]}] for _ in range(n)]
    inputs = processor.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
        padding=True).to(model.device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True,
                             temperature=temperature, top_p=1.0, top_k=0)
    comp = out[:, inputs["input_ids"].shape[1]:]
    return inputs, comp, processor.batch_decode(comp, skip_special_tokens=True)


def completion_logprobs(model, inputs, comp, mask, a, b):
    """logprob từng token trả lời của các chuỗi a..b, shape [b-a, Lc]."""
    lc = comp.shape[1]
    ids = torch.cat([inputs["input_ids"][a:b], comp[a:b]], 1)
    att = torch.cat([inputs["attention_mask"][a:b], mask[a:b]], 1)
    kw = {}
    if "mm_token_type_ids" in inputs:
        kw["mm_token_type_ids"] = torch.cat([inputs["mm_token_type_ids"][a:b], torch.zeros_like(comp[a:b])], 1)
    pv, grid = slice_vision(inputs["pixel_values"], inputs["image_grid_thw"], a, b)
    args = dict(input_ids=ids, attention_mask=att, pixel_values=pv, image_grid_thw=grid, **kw)
    try:
        out = model(logits_to_keep=lc + 1, **args)   # chỉ tính logits cho phần trả lời: đỡ tốn VRAM
    except TypeError:
        out = model(**args)
    logits = out.logits[:, -(lc + 1):-1].float()  # vị trí t dự đoán token t+1
    return torch.log_softmax(logits, -1).gather(-1, comp[a:b, :, None]).squeeze(-1)


# --------------------------------------------------------------- vòng huấn luyện
def grpo_train(model, processor, entries, cfg, out_dir, data_dir="data", max_steps=None, copy_to=None,
               save_projector=True, prompts_per_step=None, micro_batch=2, log_every=1):
    """cfg: configs/e5.yaml. model đã nạp bằng load_trainable. Trả run_info có `history` từng bước."""
    g = cfg["grpo"]
    if g.get("beta", 0.0) != 0.0:
        raise NotImplementedError("vòng lặp này chỉ hỗ trợ beta = 0 (không KL)")
    steps = max_steps or g["max_steps"]
    n, max_len, lr = g["num_generations"], g["max_completion_length"], g["learning_rate"]
    pps = prompts_per_step or g.get("prompts_per_step", 2)
    temperature = g.get("temperature", 1.0)
    seed, coord = cfg.get("seed", 42), cfg.get("coord", "norm1000")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    torch.manual_seed(seed)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    eos_ids = _eos_ids(model, processor)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    history = []

    for step in range(1, steps + 1):
        # 1) sinh và chấm điểm (chế độ eval để dùng cache, không dropout)
        model.eval()
        model.config.use_cache = True
        groups, texts_all, rewards_all = [], [], []
        for e in rng.sample(entries, pps):
            inputs, comp, texts = sample_group(model, processor, e, data_dir, coord, n, max_len, temperature)
            gt = entry_gt_boxes(e)
            rewards = [reward(t, gt) for t in texts]
            groups.append((inputs, comp, group_advantages(rewards)))
            texts_all += texts
            rewards_all += rewards

        # 2) cập nhật theo advantage (chế độ train để gradient checkpointing hoạt động)
        model.train()
        model.config.use_cache = False
        total_seqs, loss_val, used = pps * n, 0.0, 0
        for inputs, comp, adv in groups:
            if max(abs(a) for a in adv) < 1e-6:
                continue  # cả nhóm cùng reward: không có tín hiệu
            mask = completion_mask(comp, eos_ids)
            for a in range(0, n, micro_batch):
                b = min(a + micro_batch, n)
                with torch.autocast("cuda", dtype=torch.float16):
                    logp = completion_logprobs(model, inputs, comp, mask, a, b)
                m = mask[a:b].float()
                seq_logp = (logp * m).sum(1) / m.sum(1).clamp(min=1)
                adv_t = torch.tensor(adv[a:b], device=seq_logp.device)
                loss = -(adv_t * seq_logp).sum() / total_seqs
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"loss={loss.item()} ở bước {step}")
                scaler.scale(loss).backward()
                loss_val += loss.item()
                used += b - a
        if used:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
        opt.zero_grad(set_to_none=True)

        st = reward_stats(texts_all, rewards_all)
        st.update({"step": step, "loss": loss_val, "seqs_used": used,
                   "len": float(sum(len(processor.tokenizer(t)["input_ids"]) for t in texts_all) / len(texts_all)),
                   "minutes": round((time.time() - t0) / 60, 2)})
        history.append(st)
        if step % log_every == 0:
            print(f"step {step:3d}/{steps} reward={st['reward']:.3f} invalid={st['invalid']:.2f} "
                  f"empty={st['empty']:.2f} boxes={st['boxes']:.2f} len={st['len']:.0f} "
                  f"dùng {used}/{total_seqs} câu  {st['minutes']:.1f} phút", flush=True)
        if step % 20 == 0 and step != steps:
            save_checkpoint(model, out_dir / f"step{step}", save_projector, copy_to)

    run = {"name": cfg.get("name", "E5"), "steps": steps, "trainable_params": count_trainable(model),
           "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3),
           "train_minutes": round((time.time() - t0) / 60, 2), "prompts_per_step": pps,
           "final_reward": history[-1]["reward"], "history": history, "cfg": cfg}
    save_checkpoint(model, out_dir / "final", save_projector, copy_to)
    (out_dir / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    if copy_to:
        import shutil
        shutil.copy2(out_dir / "run.json", Path(copy_to) / "run.json")
    return run
