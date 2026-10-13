import re

import pytest

from src import model_utils as mu

LLM_NAMES = [f"model.language_model.layers.{i}.{s}" for i in range(36) for s in (
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")]
VIT_NAMES = [f"model.visual.blocks.{i}.{s}" for i in range(24) for s in (
    "attn.qkv", "attn.proj", "mlp.linear_fc1", "mlp.linear_fc2")]
# module KHÔNG được khớp
DISTRACTORS = ["lm_head", "model.language_model.embed_tokens", "model.language_model.layers.0.input_layernorm",
               "model.language_model.layers.0.self_attn", "model.language_model.layers.0.self_attn.q_norm",
               "model.visual.merger.linear_fc1", "model.visual.merger.linear_fc2",
               "model.visual.deepstack_merger_list.0.linear_fc1", "model.visual.patch_embed.proj",
               "model.visual.blocks.0.norm1", "model.visual.blocks.0.attn"]
ALL = LLM_NAMES + VIT_NAMES + DISTRACTORS


def count(targets):
    rx = mu.target_regex(targets)
    return sum(1 for n in ALL if re.fullmatch(rx, n))


def test_llm_regex_matches_252():
    assert count("llm") == mu.EXPECTED_MODULES["llm"] == 252


def test_llm_vit_regex_matches_348():
    assert count("llm+vit") == mu.EXPECTED_MODULES["llm+vit"] == 348


def test_llm_regex_ignores_vit_and_projector():
    rx = mu.target_regex("llm")
    assert not any(re.fullmatch(rx, n) for n in VIT_NAMES + DISTRACTORS)


def test_regex_also_matches_with_extra_prefix():
    # sau khi bọc peft hoặc ở phiên bản transformers khác, tiền tố có thể dài hơn
    assert re.fullmatch(mu.target_regex("llm"), "model.model.language_model.layers.3.mlp.up_proj")


def test_bad_targets():
    with pytest.raises(ValueError):
        mu.target_regex("vit")


def test_projector_param_detection_and_key():
    yes = ["base_model.model.model.visual.merger.linear_fc1.weight",
           "model.visual.deepstack_merger_list.2.norm.bias"]
    no = ["model.visual.blocks.0.attn.qkv.lora_A.default.weight", "model.language_model.layers.0.mlp.up_proj.weight",
          "model.visual.patch_embed.proj.weight"]
    assert all(mu.is_projector_param(n) for n in yes)
    assert not any(mu.is_projector_param(n) for n in no)
    assert mu._projector_key(yes[0]) == "visual.merger.linear_fc1.weight"


def test_quantized_modules_finds_only_4bit_layers():
    import torch

    class Linear4bit(torch.nn.Module):  # cùng tên lớp với bitsandbytes
        pass

    m = torch.nn.Module()
    m.visual = torch.nn.Module()
    m.visual.merger = torch.nn.Module()
    m.visual.merger.linear_fc1 = Linear4bit()
    m.visual.merger.norm = torch.nn.LayerNorm(4)
    m.language_model = torch.nn.Module()
    m.language_model.q_proj = Linear4bit()
    assert mu.quantized_modules(m, "visual") == ["visual.merger.linear_fc1"]
    assert mu.quantized_modules(m, "language_model") == ["language_model.q_proj"]
    assert mu.quantized_modules(m, "lm_head") == []
