#!/usr/bin/env python3
"""
losses.py ── 损失函数集合 (caption_model 包内)

包括:
  - MultiPositiveSigmoidContrastLoss: SigLIP 论文风格的多正样本 sigmoid 对比损失
  - ROIAttributeContrastLoss: 同图 ROI 内对比损失

[NAMING-NOTE]  本文件里 "SigLIP 风格" 是对 Zhai et al. 2023 论文
   "Sigmoid Loss for Language Image Pre-training" 的方法引用,
   不是本项目的字段名. 本项目实际视觉编码器是 EVA-ViT-g/14 (LAVIS 版),
   不含真正的 SigLIP encoder. 这一注释跟 [SIGLIP-LEGACY-RENAMED]
   字段重命名工作互不影响, 保留 "SigLIP" 是为了忠实保留论文出处.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# MultiPositiveSigmoidContrastLoss
# ============================================================
class MultiPositiveSigmoidContrastLoss(nn.Module):
    """
    SigLIP 论文 (Zhai et al. 2023) 风格的 sigmoid 对比损失 (允许多正样本).

    可学习的 t' (温度) 和 bias.

    Note: "SigLIP" 在此为方法出处, 不是字段名. 本项目实际视觉编码器是
    EVA-ViT-g, 没有 SigLIP encoder.
    """

    def __init__(self, init_temperature=10.0, init_bias=-10.0):
        super().__init__()
        self.t_prime = nn.Parameter(torch.tensor(math.log(init_temperature)))
        self.bias = nn.Parameter(torch.tensor(init_bias))

    def forward(self, img_feats, txt_feats, img_ids=None):
        B = img_feats.shape[0]
        T = txt_feats.shape[0]
        device = img_feats.device
        t = torch.exp(self.t_prime)
        logits = img_feats @ txt_feats.T * t + self.bias
        labels = -torch.ones(B, T, device=device)
        if img_ids is not None:
            for j in range(T):
                i = int(img_ids[j])
                if 0 <= i < B:
                    labels[i, j] = 1.0
        else:
            for i in range(min(B, T)):
                labels[i, i] = 1.0
        return -F.logsigmoid(labels * logits).sum() / B


# ============================================================
# ROIAttributeContrastLoss
# ============================================================
class ROIAttributeContrastLoss(nn.Module):
    """
    同一图像内 ROI 之间的对比损失

    对每张图,把它的 N 个 ROI 互为正样本 (除自身),
    与其他图的所有 ROI 互为负样本。
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature))

    def forward(self, roi_feats_normed):
        B, N, D = roi_feats_normed.shape
        feats = roi_feats_normed.reshape(B * N, D).float()
        sim = feats @ feats.T / self.temperature.clamp(min=0.01)
        sim.fill_diagonal_(-1e9)
        loss_list = []
        for b in range(B):
            s = b * N
            e = s + N
            sim_b = sim[s:e]
            pos_mask = torch.zeros(N, B * N, dtype=torch.bool, device=feats.device)
            pos_mask[:, s:e] = True
            for i in range(N):
                pos_mask[i, s + i] = False
            if pos_mask[0].sum() == 0:
                continue
            ls = F.log_softmax(sim_b, dim=-1)
            loss_list.append(
                -(ls * pos_mask.float()).sum(-1)
                / pos_mask.float().sum(-1).clamp(min=1)
            )
        if not loss_list:
            return torch.tensor(0.0, device=feats.device, requires_grad=True)
        return torch.stack([l.mean() for l in loss_list]).mean()