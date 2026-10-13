"""Nạp Qwen3-VL-4B (QLoRA 4-bit) và dựng setup đóng băng/LoRA cho E1, E2, E3.

Các import nặng (torch, transformers, peft) nằm trong hàm để file import được ở máy không có GPU.
"""
import re

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"

# Tên module thật của Qwen3-VL (GUIDE bước 5). Dùng ".*" ở đầu vì tiền tố khác nhau giữa các phiên bản
# transformers (model.language_model... hay model.model.language_model...).
LLM_REGEX = r".*language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
VIT_REGEX = r".*visual\.blocks\.\d+\.(attn\.(qkv|proj)|mlp\.(linear_fc1|linear_fc2))"
EXPECTED_MODULES = {"llm": 36 * 7, "llm+vit": 36 * 7 + 24 * 4}  # 252 và 348
PROJECTOR_KEYS = ("visual.merger", "visual.deepstack_merger_list")
EXTRA_FILE = "extra_trainable.pt"
# Một số bản transformers so khớp tên module theo tiền tố đầy đủ ("model.visual..."), nên "visual" một mình
# không đủ để giữ ViT ở fp16. Ghi cả hai dạng; load_model kiểm tra lại sau khi nạp.
SKIP_QUANT = ["visual", "model.visual", "lm_head"]


def target_regex(targets):
    """Regex fullmatch cho peft `target_modules`. targets: "llm" | "llm+vit"."""
    if targets == "llm":
        return LLM_REGEX
    if targets == "llm+vit":
        return f"(?:{LLM_REGEX})|(?:{VIT_REGEX})"
    raise ValueError(f"targets phải là 'llm' hoặc 'llm+vit', nhận {targets!r}")


def is_projector_param(name):
    return any(k in name for k in PROJECTOR_KEYS)


def _projector_key(name):
    """Tên tham số bỏ tiền tố peft/model: 'base_model.model.model.visual.merger.x' -> 'visual.merger.x'."""
    return name[name.index("visual."):]


def quantized_modules(model, key):
    """Tên các lớp Linear 4-bit của bitsandbytes có chứa `key` trong tên."""
    return [n for n, m in model.named_modules() if key in n and type(m).__name__ == "Linear4bit"]


def _replace_param(model, name, data, trainable):
    """Thay Parameter bằng một Parameter mới (dtype và grad_dtype mới, không dính trạng thái cũ)."""
    import torch
    module_name, _, leaf = name.rpartition(".")
    module = model.get_submodule(module_name) if module_name else model
    setattr(module, leaf, torch.nn.Parameter(data, requires_grad=trainable))


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_model(model_id=MODEL_ID):
    """Model 4-bit NF4. ViT (kể cả projector) và lm_head giữ fp16: không lượng tử hoá."""
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True, llm_int8_skip_modules=SKIP_QUANT)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, quantization_config=quant, dtype=torch.float16, device_map={"": 0})
    # ViT/projector bị lượng tử hoá thì E2/E3 không huấn luyện đầy đủ projector được (và E0 lệch đề): dừng ngay
    bad = quantized_modules(model, "visual") + quantized_modules(model, "lm_head")
    if bad:
        raise RuntimeError(f"{len(bad)} lớp trong ViT/lm_head bị lượng tử hoá 4-bit, ví dụ {bad[:3]}. "
                           f"Sửa SKIP_QUANT trong src/model_utils.py cho khớp tên module.")
    print(f"4-bit: {len(quantized_modules(model, 'language_model'))} lớp Linear của LLM; ViT và lm_head giữ fp16")
    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "left"  # cần cho generate theo batch
    return model, processor


def apply_setup(model, targets="llm", train_projector=False, r=16, alpha=None, dropout=0.05, strict=True):
    """Đóng băng toàn bộ, gắn LoRA, và (E2/E3) mở đầy đủ projector ở float32.

    Trả (peft_model, info). info["matched_modules"] phải là 252 (E1, E2) hoặc 348 (E3).
    """
    import torch
    from peft import LoraConfig, get_peft_model

    regex = target_regex(targets)
    matched = [n for n, _ in model.named_modules() if re.fullmatch(regex, n)]
    if strict and len(matched) != EXPECTED_MODULES[targets]:
        hint = [n for n, _ in model.named_modules() if "layers.0." in n or "blocks.0." in n][:12]
        raise ValueError(f"regex khớp {len(matched)} module, cần {EXPECTED_MODULES[targets]}. "
                         f"Tên module mẫu để chỉnh regex: {hint}")

    for p in model.parameters():
        p.requires_grad = False
    model.enable_input_require_grads()  # cần cho gradient checkpointing

    cfg = LoraConfig(r=r, lora_alpha=alpha or 2 * r, lora_dropout=dropout, target_modules=regex, bias="none")
    peft_model = get_peft_model(model, cfg)

    # get_peft_model đóng băng mọi thứ ngoài adapter, nên phải mở projector SAU bước này.
    n_proj = 0
    if train_projector:
        for name, p in list(peft_model.named_parameters()):
            if is_projector_param(name):
                if not p.is_floating_point():  # trọng số 4-bit đã nén: đổi sang float chỉ ra số vô nghĩa
                    raise RuntimeError(f"{name} bị lượng tử hoá ({p.dtype}): không huấn luyện đầy đủ được")
                _replace_param(peft_model, name, p.detach().float(), trainable=True)
                n_proj += p.numel()
    # GradScaler.unscale_ từ chối gradient fp16: mọi tham số huấn luyện phải là fp32
    for name, p in list(peft_model.named_parameters()):
        if p.requires_grad and p.dtype != torch.float32:
            _replace_param(peft_model, name, p.detach().float(), trainable=True)
    bad = [n for n, p in peft_model.named_parameters() if p.requires_grad and p.dtype != torch.float32]
    if bad:
        raise RuntimeError(f"{len(bad)} tham số huấn luyện vẫn không phải fp32: {bad[:3]}")

    info = {"targets": targets, "train_projector": train_projector, "r": r, "alpha": alpha or 2 * r,
            "matched_modules": len(matched), "projector_params": n_proj,
            "trainable_params": count_trainable(peft_model)}
    return peft_model, info


def apply_setup_from_cfg(model, cfg):
    lora = cfg["lora"]
    return apply_setup(model, targets=lora["targets"], train_projector=cfg["train_projector"],
                       r=lora["r"], alpha=lora.get("alpha"), dropout=lora.get("dropout", 0.05))


def save_extra(model, path):
    """Lưu riêng projector đã huấn luyện (peft.save_pretrained KHÔNG lưu chúng)."""
    import torch
    state = {_projector_key(n): p.detach().float().cpu()
             for n, p in model.named_parameters() if is_projector_param(n)}
    if not state:
        raise ValueError("không tìm thấy tham số projector để lưu")
    torch.save(state, path)
    return len(state)


def load_extra(model, path):
    """Nạp lại projector đã huấn luyện, giữ fp32 như lúc train (chạy generate trong autocast fp16)."""
    import torch
    saved = torch.load(path, map_location="cpu")
    names = {_projector_key(n): n for n, _ in model.named_parameters() if is_projector_param(n)}
    if set(saved) != set(names):
        raise ValueError(f"projector không khớp: thiếu {sorted(set(names) - set(saved))[:3]}, "
                         f"thừa {sorted(set(saved) - set(names))[:3]}")
    device = next(model.parameters()).device
    for k, v in saved.items():
        _replace_param(model, names[k], v.to(device, torch.float32), trainable=False)
    return len(saved)
