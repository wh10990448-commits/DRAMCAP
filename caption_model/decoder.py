#!/usr/bin/env python3
"""
decoder.py v6.5 ── MistralDecoder (LLM 解码器)

[V6.5-PROMPT] 默认 (无文字 marker): <BOS><SYS><vis_32><name_8><attr_6><INSTRUCT>
              name 和 attr 直接连续放在 vis 后面, LLM 不需要也无法区分两者
[V6.4-PROMPT] 旧格式 (有文字 marker, 已弃用): vis + " Objects: " + name + " Attributes: " + attr
"""

import os
import torch
import torch.nn as nn

from transformers import MistralForCausalLM

from .constants import QFORMER_HIDDEN, LLM_HIDDEN
from .auxiliary import EVCapVisualProjector
from .utils import patch_torch_load_safety

try:
    from peft import get_peft_model, LoraConfig, TaskType
    _PEFT_AVAILABLE = True
except ImportError:
    _PEFT_AVAILABLE = False


class MistralDecoder(nn.Module):
    """
    Mistral / Llama LLM 解码器

    特性:
      - LoRA 兼容 (PEFT)
      - 视觉投影 vis_proj_A / vis_proj_B
      - prompt 模板可切换 stage1 / stage2_early / stage2_late
      - [V6.3-PROMPT]: 新格式 ' Objects: ' / ' Attributes: ' marker
      - [V6.3-LEGACY]: _legacy_prompt_v62=True 时回退老格式
    """

    _SYSTEM = ("A chat between a curious user and an artificial intelligence assistant. "
               "The assistant gives helpful, detailed, and polite answers to the user's questions. ")
    _USER_HEAD = "USER: "
    _ASSISTANT_HEAD = " ASSISTANT:"
    _INSTRUCT_TEMPLATES = {
        "stage1":       "\nDescribe the image in one sentence.",
        "stage2_early": "\nDescribe the image in one sentence.",
        "stage2_late":  "\nDescribe the image in one sentence.",
    }

    def __init__(self, llm_path=None):
        super().__init__()
        patch_torch_load_safety()

        # 自动检测模型类型 (mistral / llama)
        import json as _json
        _config_path = os.path.join(llm_path, "config.json") if llm_path else ""
        _model_type = "unknown"
        if os.path.exists(_config_path):
            with open(_config_path) as _cf:
                _model_type = _json.load(_cf).get("model_type", "unknown")
        if _model_type == "llama":
            from transformers import LlamaForCausalLM
            self.llm = LlamaForCausalLM.from_pretrained(
                llm_path, torch_dtype=torch.bfloat16, device_map=None)
        else:
            self.llm = MistralForCausalLM.from_pretrained(
                llm_path, torch_dtype=torch.bfloat16, device_map=None)

        # 关闭 gradient checkpointing
        try:
            if hasattr(self.llm, 'gradient_checkpointing_disable'):
                self.llm.gradient_checkpointing_disable()
            for _m in self.llm.modules():
                if hasattr(_m, 'gradient_checkpointing'):
                    _m.gradient_checkpointing = False
            if hasattr(self.llm, 'config'):
                self.llm.config.use_cache = False
            _gc_status = getattr(self.llm, 'is_gradient_checkpointing', None)
            print(f"  [GC-OFF] llm.gradient_checkpointing_disable() 已调用")
            print(f"  [GC-CHECK] is_gradient_checkpointing = {_gc_status}")
        except Exception as _e:
            print(f"  [GC-OFF] ⚠️  关闭 gradient checkpointing 失败: {_e}")

        for p in self.llm.parameters():
            p.requires_grad = False
        self._lora_enabled = False
        self._model_type = _model_type
        self._prompt_stage = "stage2_late"

        # ★ [V6.5-FIX] 默认改为 v6.5 格式 (无文字 marker, 但保留 attr 段)
        # True (默认 v6.5):  <BOS><SYS><vis><name_8><attr_6><INSTRUCT>  无 Objects:/Attributes: marker
        # False (旧 v6.4):   <BOS><SYS><vis> Objects:<name> Attributes:<attr><INSTRUCT>
        # 因为 LLM 看不出哪些 token 是 name 哪些是 attr, 加文字 marker 反而是噪声
        self._legacy_prompt_v62 = True

        llm_hidden = self.llm.config.hidden_size
        self.vocab_size = self.llm.config.vocab_size
        self.vis_proj_A = EVCapVisualProjector(
            vis_dim=QFORMER_HIDDEN, llm_hidden=llm_hidden)
        self.vis_proj_B = EVCapVisualProjector(
            vis_dim=QFORMER_HIDDEN, llm_hidden=llm_hidden)
        print(f"  LLM: {_model_type}, hidden={llm_hidden}")
        self._tokenizer = None

    # ------------------------------------------------------------
    # 公共属性 / 方法
    # ------------------------------------------------------------
    @property
    def vis_proj(self):
        return self.vis_proj_A

    def set_tokenizer(self, tok):
        self._tokenizer = tok

    def _get_embed_fn(self):
        return self.llm.get_input_embeddings()

    def enable_lora(self):
        if not _PEFT_AVAILABLE or self._lora_enabled:
            return
        self.llm = get_peft_model(
            self.llm,
            LoraConfig(
                r=16, lora_alpha=32,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                task_type=TaskType.CAUSAL_LM,
            ),
        )
        self._lora_enabled = True

    def set_prompt_stage(self, stage):
        self._prompt_stage = stage

    @torch.no_grad()
    def _encode_text(self, text, device):
        ids = self._tokenizer.encode(
            text, add_special_tokens=False, return_tensors='pt').to(device)
        return self._get_embed_fn()(ids).to(torch.bfloat16), ids.shape[1]

    # ------------------------------------------------------------
    # Prompt 拼接核心逻辑
    # ------------------------------------------------------------
    def _build_prompt_embeds(self, B, device, vis_embeds,
                             name_embeds=None, attr_embeds=None,
                             name_attn=None, attr_attn=None):
        """
        [V6.5-PROMPT] 默认格式 (self._legacy_prompt_v62=True):
          BOS + SYS + vis_32q + name_8 + attr_6 + INSTRUCT
          → 共 14 个视觉 token (前 8 name + 后 6 attr) 直接连续, 无文字 marker
          → LLM 看到的是连续视觉 token 串, 不需要也无法区分 name/attr

        [V6.4-PROMPT] 旧格式 (self._legacy_prompt_v62=False):
          BOS + SYS + vis_32q + " Objects: " + name_K + " Attributes: " + attr_M + INSTRUCT
        """
        # [V6.5-FIX] legacy 模式保留 attr_embeds, 只是不加 "Objects:"/"Attributes:" 文字 marker
        # (旧版 legacy 会丢弃 attr, 现在保留 — 这样属性信息不丢)

        if self._tokenizer is None:
            parts = [vis_embeds]
            if name_embeds is not None:
                parts.append(name_embeds)
            if attr_embeds is not None:
                parts.append(attr_embeds)
            total = torch.cat(parts, dim=1)
            attn = torch.ones(B, total.shape[1], dtype=torch.long, device=device)
            return total, attn, total.shape[1]

        dtype = vis_embeds.dtype
        bos_id = self._tokenizer.bos_token_id or 1
        bos_emb = self._get_embed_fn()(
            torch.tensor([[bos_id]], device=device)
        ).to(dtype).expand(B, -1, -1)
        sys_emb, _ = self._encode_text(self._SYSTEM + self._USER_HEAD, device)
        sys_emb = sys_emb.expand(B, -1, -1)

        # 文字 marker (仅 v6.4 旧格式使用, v6.5 默认不用)
        if not self._legacy_prompt_v62:
            obj_marker_emb, _ = self._encode_text(" Objects: ", device)
            attr_marker_emb, _ = self._encode_text(" Attributes: ", device)
            obj_marker_emb = obj_marker_emb.expand(B, -1, -1)
            attr_marker_emb = attr_marker_emb.expand(B, -1, -1)

        inst_text = self._INSTRUCT_TEMPLATES.get(
            self._prompt_stage or "stage2_late",
            self._INSTRUCT_TEMPLATES["stage2_late"]) + self._ASSISTANT_HEAD
        inst_emb, _ = self._encode_text(inst_text, device)
        inst_emb = inst_emb.expand(B, -1, -1)

        parts = [bos_emb, sys_emb, vis_embeds]
        attn_parts = [
            torch.ones(B, bos_emb.shape[1], dtype=torch.long, device=device),
            torch.ones(B, sys_emb.shape[1], dtype=torch.long, device=device),
            torch.ones(B, vis_embeds.shape[1], dtype=torch.long, device=device),
        ]

        if name_embeds is not None:
            # v6.4 旧格式: 在 name 前加 "Objects:" marker
            if not self._legacy_prompt_v62:
                parts.append(obj_marker_emb)
                attn_parts.append(torch.ones(
                    B, obj_marker_emb.shape[1], dtype=torch.long, device=device))
            parts.append(name_embeds)
            if name_attn is None:
                name_attn_use = torch.ones(
                    B, name_embeds.shape[1], dtype=torch.long, device=device)
            else:
                name_attn_use = name_attn.long()
            attn_parts.append(name_attn_use)

        # [V6.5-FIX] attr 段: legacy(默认) 也保留 attr_embeds, 只是不加 "Attributes:" marker
        if attr_embeds is not None:
            if not self._legacy_prompt_v62:
                parts.append(attr_marker_emb)
                attn_parts.append(torch.ones(
                    B, attr_marker_emb.shape[1], dtype=torch.long, device=device))
            parts.append(attr_embeds)
            if attr_attn is None:
                attr_attn_use = torch.ones(
                    B, attr_embeds.shape[1], dtype=torch.long, device=device)
            else:
                attr_attn_use = attr_attn.long()
            attn_parts.append(attr_attn_use)

        parts.append(inst_emb)
        attn_parts.append(torch.ones(
            B, inst_emb.shape[1], dtype=torch.long, device=device))

        prefix_emb = torch.cat(parts, dim=1)
        prefix_attn = torch.cat(attn_parts, dim=1)
        return prefix_emb, prefix_attn, prefix_emb.shape[1]

    # ------------------------------------------------------------
    # Forward (训练)
    # ------------------------------------------------------------
    def forward(self, vis_tokens_768, text_tokens, text_mask=None,
                name_tokens_768=None, attr_tokens_768=None,
                name_mask=None, attr_mask=None, **kwargs):
        B = vis_tokens_768.shape[0]
        device = vis_tokens_768.device
        vis_embeds = self.vis_proj_A(vis_tokens_768).to(torch.bfloat16)
        name_embeds = (self.vis_proj_A(name_tokens_768).to(torch.bfloat16)
                       if name_tokens_768 is not None else None)
        attr_embeds = (self.vis_proj_A(attr_tokens_768).to(torch.bfloat16)
                       if attr_tokens_768 is not None else None)

        name_attn = name_mask.long() if name_mask is not None else None
        attr_attn = attr_mask.long() if attr_mask is not None else None

        prefix_emb, prefix_attn, n_prefix = self._build_prompt_embeds(
            B, device, vis_embeds, name_embeds, attr_embeds,
            name_attn=name_attn, attr_attn=attr_attn)

        txt_input = text_tokens[:, :-1]
        txt_target = text_tokens[:, 1:]
        L_txt = txt_input.shape[1]
        txt_embeds = self._get_embed_fn()(txt_input)
        txt_attn = (text_mask[:, :-1].long() if text_mask is not None
                    else torch.ones(B, L_txt, dtype=torch.long, device=device))
        pad_id = self._tokenizer.pad_token_id if self._tokenizer else 2
        txt_target = txt_target.clone()
        txt_target[txt_target == pad_id] = -100
        inputs_embeds = torch.cat([prefix_emb, txt_embeds], dim=1)
        full_attn = torch.cat([prefix_attn, txt_attn], dim=1)
        ignore_prefix = torch.full((B, n_prefix), -100, dtype=torch.long, device=device)
        labels = torch.cat([ignore_prefix, txt_target], dim=1)
        return self.llm(
            inputs_embeds=inputs_embeds, attention_mask=full_attn,
            labels=labels, return_dict=True,
        ).logits

    # ------------------------------------------------------------
    # Generate (推理)
    # ------------------------------------------------------------
    @torch.no_grad()
    def generate(self, vis_tokens_768, name_tokens_768=None, attr_tokens_768=None,
                 name_mask=None, attr_mask=None, **kwargs):
        B = vis_tokens_768.shape[0]
        device = vis_tokens_768.device
        vis_embeds = self.vis_proj_A(vis_tokens_768).to(torch.bfloat16)
        name_embeds = (self.vis_proj_A(name_tokens_768).to(torch.bfloat16)
                       if name_tokens_768 is not None else None)
        attr_embeds = (self.vis_proj_A(attr_tokens_768).to(torch.bfloat16)
                       if attr_tokens_768 is not None else None)

        name_attn = name_mask.long() if name_mask is not None else None
        attr_attn = attr_mask.long() if attr_mask is not None else None

        prefix_emb, prefix_attn, _ = self._build_prompt_embeds(
            B, device, vis_embeds, name_embeds, attr_embeds,
            name_attn=name_attn, attr_attn=attr_attn)

        eos_id = self._tokenizer.eos_token_id if self._tokenizer else 2
        stop_ids = [eos_id]
        if self._tokenizer:
            nl = self._tokenizer.encode("\n", add_special_tokens=False)
            if nl:
                stop_ids.append(nl[0])
        gen_kwargs = dict(
            max_new_tokens=30, min_new_tokens=6, num_beams=5,
            length_penalty=1.0, early_stopping=True, no_repeat_ngram_size=3,
            eos_token_id=stop_ids, pad_token_id=eos_id,
        )
        # 清除 caller 可能传进来的不相关 kwargs
        for k in (
            'category_ids', 'tag_sequences', 'is_core', 'importance_scores',
            'object_names', 'use_reranking', 'clip_text_feat', 'tag_confidence',
            'name_tokens', 'name_tokens_768', 'attr_tokens', 'attr_tokens_768',
            'name_mask', 'attr_mask',
        ):
            kwargs.pop(k, None)
        gen_kwargs.update(kwargs)
        return self.llm.generate(
            inputs_embeds=prefix_emb, attention_mask=prefix_attn, **gen_kwargs)
