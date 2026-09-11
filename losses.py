#!/usr/bin/env python3
"""
losses.py v6.0+eva — 精简版：仅 NTP 损失  [SIGLIP-LEGACY-RENAMED]

适配: model.py (LAVIS EVA-ViT-g 视觉前端 + eva_* 命名) + train.py v6.0+

[SIGLIP-LEGACY-RENAMED] 改动 (相对旧版):
  全工程把误导性的 siglip_* / clip_* 统一改为 eva_*. 本工程只有单一
  EVA-CLIP ViT-g 视觉编码器, 并无 SigLIP / 第二个 CLIP. 本文件测试代码
  里的辅助字段相应更新, 跟 model.py forward(return_intermediates=True)
  实际返回的字段名对齐:

      siglip_g_for_loss   →  eva_global_768_for_loss   (model.py 实际返回名)
      roi_processed       →  roi_processed_for_loss    (与 model.py 对齐)

  ★ 重要: CaptionLoss 是纯 NTP (Image-grounded Text Generation loss),
    上述视觉字段全部经 **kwargs 静默忽略, 故改名不影响任何损失计算
    (loss 与视觉字段命名完全解耦, 改完不需要重训).

继承 v5.4.1:
  [FIX-LS]  CaptionLoss.set_label_smoothing() 方法保留
            train.py 的 _maybe_update_label_smoothing() 需要调用此方法
  [FIX-LS]  暴露 label_smoothing 属性

设计原则:
  辅助损失的职责已明确分配给其他模块, 此处只做 NTP:
    - 对齐 EVCap 原版 ITG (Image-grounded Text Generation) loss
    - 不含 contrast / ret_align / sigmoid / roi_attr 等辅助损失
      (sigmoid contrastive / ROI attribute contrast 这两个在
       caption_model/losses.py 里, 由 model.compute_auxiliary_losses 调用,
       跟本文件无关)
    - 默认 label_smoothing=0.0 (对齐 EVCap 原版)
    - 空间位置偏置 (ROITokenAggregator 的 pos_bias) 经标准 NTP 梯度训练,
      不需要任何独立损失项 → 本文件无需为其改动

  CaptionLoss 保持单一职责: Next Token Prediction.
  兼容旧接口 CaptionLoss(w1=1.0, w2=0.0, ...), 多余参数通过 **kwargs 静默忽略.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional


class NTPLoss(nn.Module):
    """标准自回归语言模型损失"""

    def __init__(self, ignore_index: int = -100, label_smoothing: float = 0.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.ce = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
        )

    def set_label_smoothing(self, value: float):
        """动态更新 label smoothing 值"""
        self.label_smoothing = value
        self.ce = nn.CrossEntropyLoss(
            ignore_index=self.ignore_index,
            label_smoothing=value,
        )

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, T, V) 模型输出 logits
            labels: (B, T)    目标 token ids, padding 位置 = -100
        Returns:
            scalar loss
        """
        B, T, V = logits.shape
        loss = self.ce(logits.reshape(B * T, V), labels.reshape(B * T))
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.zeros_like(loss)
        return loss


class CaptionLoss(nn.Module):
    """
    精简版统一损失: 仅 NTP (Image-grounded Text Generation loss).

    兼容旧接口:
        CaptionLoss(w1=1.0, w2=0.0, w3=0.0, w4=0.0, ignore_index=-100)
    新接口:
        CaptionLoss(ignore_index=-100, label_smoothing=0.0)

    多余的 w2/w3/w4/lambda_sigmoid/lambda_roi_attr 等参数静默忽略.
    forward() 里所有非 logits/labels 的 kwargs 也静默忽略
    (包括 eva_global_768_for_loss / roi_processed_for_loss 等视觉字段).
    """

    def __init__(
        self,
        w1: float = 1.0,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.w_ntp = w1
        self.label_smoothing = label_smoothing
        self.ntp_loss = NTPLoss(ignore_index, label_smoothing)

    def set_label_smoothing(self, value: float):
        """[FIX-LS] 动态更新 label smoothing 值, 供 train.py LS-CUR 调用"""
        self.label_smoothing = value
        self.ntp_loss.set_label_smoothing(value)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            logits: (B, T, V) decoder logits
            labels: (B, T)    目标 token ids
            **kwargs:         忽略所有其他参数 (兼容旧调用 + 视觉字段)
        Returns:
            (total_loss, details_dict)
        """
        ntp = self.ntp_loss(logits, labels)
        total = self.w_ntp * ntp
        details = {
            'NTPLoss': float(ntp.item()),
            'total':   float(total.item()),
        }
        return total, details


# ==============================================================================
# 测试
# ==============================================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    B, T, V = 4, 20, 32000
    logits = torch.randn(B, T, V, requires_grad=True)
    labels = torch.randint(0, V, (B, T))
    labels[:, -5:] = -100

    # 旧接口兼容测试
    print("=== 旧接口 CaptionLoss(w1=1.0, w2=0.0, w3=0.0, w4=0.0) ===")
    loss_fn = CaptionLoss(w1=1.0, w2=0.0, w3=0.0, w4=0.0, ignore_index=-100)
    total, details = loss_fn(logits=logits, labels=labels)
    print(f"  NTP={details['NTPLoss']:.4f}, total={total.item():.4f}")
    total.backward()
    print("  ✅ 旧接口通过")

    # 新接口测试
    print("\n=== 新接口 CaptionLoss(ignore_index=-100) ===")
    logits2 = torch.randn(B, T, V, requires_grad=True)
    loss_fn2 = CaptionLoss(ignore_index=-100, label_smoothing=0.0)
    total2, details2 = loss_fn2(logits2, labels)
    print(f"  NTP={details2['NTPLoss']:.4f}, total={total2.item():.4f}")
    total2.backward()
    print("  ✅ 新接口通过")

    # set_label_smoothing 测试
    print("\n=== set_label_smoothing 测试 ===")
    loss_fn3 = CaptionLoss(ignore_index=-100, label_smoothing=0.05)
    assert loss_fn3.label_smoothing == 0.05, "初始值错误"
    loss_fn3.set_label_smoothing(0.1)
    assert loss_fn3.label_smoothing == 0.1, "更新后值错误"
    assert loss_fn3.ntp_loss.label_smoothing == 0.1, "NTPLoss 值未同步"
    print(f"  label_smoothing: 0.05 → 0.1 ✅")

    # 带多余参数的兼容测试 (★ 字段名跟 model.py forward 返回的对齐)
    print("\n=== 多余参数兼容 (含 eva_* 视觉字段) ===")
    loss_fn4 = CaptionLoss(w1=1.0, w2=0.1, w3=0.05, w4=0.1,
                            ignore_index=-100, lambda_sigmoid=0.1,
                            lambda_roi_attr=0.05, stage1_epochs=0, decay_epochs=2)
    logits4 = torch.randn(B, T, V, requires_grad=True)
    total4, details4 = loss_fn4(
        logits=logits4, labels=labels,
        # ★ 这些字段名跟 model.py forward(return_intermediates=True) 返回的对齐
        # 旧名 (siglip_g_for_loss / vis_g_for_loss / roi_processed) 都被 **kwargs
        # 静默忽略, 改名前后行为完全一致
        eva_global_768_for_loss=torch.randn(B, 768),
        roi_processed_for_loss=torch.randn(B, 9, 768),
        llip_global_normed=torch.randn(B, 1408),
    )
    print(f"  NTP={details4['NTPLoss']:.4f}, total={total4.item():.4f}")
    total4.backward()
    print("  ✅ 多余参数兼容通过")

    # 旧字段名也能兼容 (确认 **kwargs 真的吃掉一切)
    print("\n=== 旧字段名 (siglip_*) 也能兼容 ===")
    logits5 = torch.randn(B, T, V, requires_grad=True)
    total5, _ = loss_fn4(
        logits=logits5, labels=labels,
        siglip_g_for_loss=torch.randn(B, 768),   # 旧名, 应被忽略
        roi_processed=torch.randn(B, 9, 768),    # 旧名, 应被忽略
    )
    total5.backward()
    print("  ✅ 旧字段名兼容 (**kwargs 静默忽略)")

    print("\n✅ 全部测试通过")