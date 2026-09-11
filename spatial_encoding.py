"""
空间编码模块 - 按照设计文档规范实现

★ V6.6.3 [FIX]: forward 里 rel_pos.view(-1, D) → .reshape(-1, D)
  原因: model.py 切片 rel_pos[:, :N_roi, :N_roi] 后内存非连续,
        .view 要求连续内存会崩, .reshape 自动处理非连续 (必要时拷贝).

绝对位置 (10维):
[0] cx: 中心 X
[1] cy: 中心 Y
[2] w: 宽度
[3] h: 高度
[4] area: w × h
[5] region: floor(cx*3) + floor(cy*3)*3 (0-8的整数，代表九宫格) / 8 归一化
[6] dist_center: 到图中心(0.5, 0.5)的欧氏距离
[7] aspect_ratio: w / h
[8] coverage_conf: 面积与中心距离的乘积（越靠近中心且越大，分值越高）
[9] is_large: area > 0.05 ? 1 : 0

相对位置 (12维) - 选取距离中心最近的物体作为Ref（参照物）:
[0] dx: cx - cx_ref
[1] dy: cy - cy_ref
[2] dist: 到参照物的中心距离
[3] sin: 相对角度的正弦值 (dy/dist)
[4] cos: 相对角度的余弦值 (dx/dist)
[5] size_log: log(area / area_ref)
[6] iou: 与参照物的交并比
[7] edge_dist: 距离画面四条边最近的那条边的距离
[8] x_overlap: X轴方向的投影重叠度
[9] y_overlap: Y轴方向的投影重叠度
[10] ref_id: 参照物的索引（归一化到0-1）
[11] diff: 中心点X轴坐标的原始差值
"""

import math
import torch
import numpy as np
import torch.nn as nn
from typing import List, Tuple


class SpatialEncoder(nn.Module):
    """空间特征编码器 - 绝对位置 + 相对位置融合"""

    def __init__(self, absolute_dim=10, relative_dim=12, hidden_dim=256, output_dim=768):
        super().__init__()
        # 绝对位置编码: [10] → [hidden] → [output]
        self.abs_enc = nn.Sequential(
            nn.Linear(absolute_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )
        # 相对位置编码: [12] → [hidden] → [output]
        self.rel_enc = nn.Sequential(
            nn.Linear(relative_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, abs_pos, rel_pos):
        """
        Args:
            abs_pos: [B, N, 10] 绝对位置特征
            rel_pos: [B, N, N, 12] 相对位置特征

        Returns:
            spatial_tokens: [B, N, output_dim]
        """
        # 绝对位置编码
        abs_encoded = self.abs_enc(abs_pos)  # [B, N, D]

        # 相对位置编码 - 平均聚合
        B, N, _, D = rel_pos.shape
        # ★ V6.6.3 [FIX]: .view → .reshape (model.py 切片后 rel_pos 可能非连续)
        rel_flat = rel_pos.reshape(-1, D)  # [B*N*N, 12]
        rel_encoded = self.rel_enc(rel_flat).reshape(B, N, N, -1)  # [B, N, N, D]
        rel_mean = rel_encoded.mean(dim=2)  # [B, N, D] 每个物体与其他所有物体关系的平均

        # 融合
        return self.norm(abs_encoded + rel_mean)


def compute_absolute_position(bbox: Tuple[int, int, int, int], W: int, H: int) -> np.ndarray:
    """
    计算单个物体的绝对位置特征 (10维)

    Args:
        bbox: (x1, y1, x2, y2) 像素坐标
        W, H: 图像宽高

    Returns:
        [10] 绝对位置编码
    """
    x1, y1, x2, y2 = bbox

    # 归一化坐标
    x1n, y1n = x1 / W, y1 / H
    x2n, y2n = x2 / W, y2 / H

    # [0-3] 基础几何
    cx = (x1n + x2n) / 2  # 中心X
    cy = (y1n + y2n) / 2  # 中心Y
    w = x2n - x1n  # 宽度
    h = y2n - y1n  # 高度

    # [4] 面积
    area = w * h

    # [5] 9宫格区域 (0-8)
    # 判定策略: 小物体用中心点，大物体用覆盖最大区域
    if area < 0.05:
        # 小物体: 直接用中心点判定
        col = int(cx * 3)
        row = int(cy * 3)
    else:
        # 大物体: 计算覆盖最大的区域
        max_coverage = 0
        best_region = 4  # 默认中心
        for r in range(3):
            for c in range(3):
                # 区域边界
                rx1, rx2 = c / 3, (c + 1) / 3
                ry1, ry2 = r / 3, (r + 1) / 3
                # 计算重叠
                ox1 = max(x1n, rx1)
                oy1 = max(y1n, ry1)
                ox2 = min(x2n, rx2)
                oy2 = min(y2n, ry2)
                ow = max(0, ox2 - ox1)
                oh = max(0, oy2 - oy1)
                coverage = ow * oh
                if coverage > max_coverage:
                    max_coverage = coverage
                    col, row = c, r

    col = min(max(col, 0), 2)
    row = min(max(row, 0), 2)
    region = row * 3 + col  # 0-8

    # [6] 到图像中心的距离
    dist_center = math.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2)

    # [7] 宽高比
    aspect_ratio = min(w / (h + 1e-6), 5.0)  # 限制最大值

    # [8] coverage_conf: 面积与中心距离的乘积（越靠近中心且越大，分值越高）
    # 修正: (1 - dist_center) 越靠近中心越大
    coverage_conf = area * (1 - dist_center)

    # [9] 是否大物体
    is_large = 1.0 if area >= 0.05 else 0.0

    return np.array([
        cx,  # [0] 中心X
        cy,  # [1] 中心Y
        w,  # [2] 宽度
        h,  # [3] 高度
        area,  # [4] 面积
        region / 8,  # [5] 9宫格区域 (归一化)
        dist_center,  # [6] 到中心距离
        aspect_ratio,  # [7] 宽高比
        coverage_conf,  # [8] 覆盖置信度
        is_large  # [9] 是否大物体
    ], dtype=np.float32)


def compute_relative_position(bbox_a: Tuple, bbox_b: Tuple, W: int, H: int,
                              idx_a: int = 0, idx_b: int = 1, N: int = 9) -> np.ndarray:
    """
    计算两个物体之间的相对位置特征 (12维)

    选取距离中心最近的物体作为参照物 (Ref)

    Args:
        bbox_a, bbox_b: (x1, y1, x2, y2)
        idx_a, idx_b: 物体索引
        N: 总物体数 (用于归一化ref_id)

    Returns:
        [12] 相对位置编码
    """
    # 归一化
    a = [bbox_a[0] / W, bbox_a[1] / H, bbox_a[2] / W, bbox_a[3] / H]
    b = [bbox_b[0] / W, bbox_b[1] / H, bbox_b[2] / W, bbox_b[3] / H]

    # 中心点
    cx_a, cy_a = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    cx_b, cy_b = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2

    # 面积
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])

    # 到图像中心的距离
    dist_a_center = math.sqrt((cx_a - 0.5) ** 2 + (cy_a - 0.5) ** 2)
    dist_b_center = math.sqrt((cx_b - 0.5) ** 2 + (cy_b - 0.5) ** 2)

    # 确定参照物: 更靠近中心的是参照物
    if dist_b_center < dist_a_center:
        # B是参照物，计算A相对于B的位置
        ref_flag = 1.0
        ref_id = idx_b / max(N - 1, 1)  # 归一化
        dx = cx_a - cx_b
        dy = cy_a - cy_b
        size_log = math.log(area_a / (area_b + 1e-6) + 1)
    else:
        # A是参照物，计算B相对于A的位置
        ref_flag = 0.0
        ref_id = idx_a / max(N - 1, 1)
        dx = cx_b - cx_a
        dy = cy_b - cy_a
        size_log = math.log(area_b / (area_a + 1e-6) + 1)

    # [2] 距离
    dist = math.sqrt(dx ** 2 + dy ** 2)

    # [3-4] 角度 (sin/cos)
    if dist > 1e-6:
        sin_ang = dy / dist
        cos_ang = dx / dist
    else:
        sin_ang, cos_ang = 0.0, 1.0

    # [6] IoU
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter_area = iw * ih
    union_area = area_a + area_b - inter_area
    iou = inter_area / (union_area + 1e-6)

    # [7] edge_dist: 距离画面四条边最近的那条边的距离
    # 使用A物体的边界距离（主体物体）
    if dist_b_center < dist_a_center:
        # A是主体
        edge_left = a[0]
        edge_right = 1 - a[2]
        edge_top = a[1]
        edge_bottom = 1 - a[3]
    else:
        # B是主体
        edge_left = b[0]
        edge_right = 1 - b[2]
        edge_top = b[1]
        edge_bottom = 1 - b[3]
    edge_dist = min(edge_left, edge_right, edge_top, edge_bottom)

    # [8-9] X/Y轴投影重叠度
    w_a, w_b = a[2] - a[0], b[2] - b[0]
    h_a, h_b = a[3] - a[1], b[3] - b[1]
    x_overlap = iw / (min(w_a, w_b) + 1e-6) if iw > 0 else 0.0
    y_overlap = ih / (min(h_a, h_b) + 1e-6) if ih > 0 else 0.0

    # [11] diff: 中心点X轴坐标的原始差值
    diff = cx_a - cx_b

    return np.array([
        dx,  # [0] X偏移
        dy,  # [1] Y偏移
        dist,  # [2] 中心距离
        sin_ang,  # [3] 角度sin
        cos_ang,  # [4] 角度cos
        size_log,  # [5] 大小比对数
        iou,  # [6] IoU
        edge_dist,  # [7] 边缘距离
        x_overlap,  # [8] X重叠
        y_overlap,  # [9] Y重叠
        ref_id,  # [10] 参照物索引 (归一化)
        diff  # [11] X差值
    ], dtype=np.float32)


def compute_spatial_features(bboxes: List[Tuple], W: int, H: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算所有物体的空间特征

    Args:
        bboxes: [(x1,y1,x2,y2), ...] N个bbox
        W, H: 图像尺寸

    Returns:
        abs_pos: [N, 10] 绝对位置
        rel_pos: [N, N, 12] 相对位置
    """
    N = len(bboxes)

    # 绝对位置
    abs_pos = np.array([compute_absolute_position(b, W, H) for b in bboxes], dtype=np.float32)

    # 相对位置 (N×N矩阵)
    rel_pos = np.zeros((N, N, 12), dtype=np.float32)
    for i in range(N):
        for j in range(N):
            if i != j:
                rel_pos[i, j] = compute_relative_position(bboxes[i], bboxes[j], W, H, i, j, N)

    return abs_pos, rel_pos


# 测试
if __name__ == "__main__":
    # 模拟场景
    W, H = 640, 480
    bboxes = [
        (50, 200, 150, 350),  # dog (左侧)
        (300, 100, 450, 400),  # person (中央)
        (500, 300, 600, 450),  # chair (右下)
    ]

    abs_pos, rel_pos = compute_spatial_features(bboxes, W, H)

    print("绝对位置 [N, 10]:")
    print(f"  Shape: {abs_pos.shape}")
    for i, name in enumerate(['dog', 'person', 'chair']):
        print(f"  {name}: region={int(abs_pos[i, 5] * 8)}, dist_center={abs_pos[i, 6]:.3f}, is_large={abs_pos[i, 9]}")

    print("\n相对位置 [N, N, 12]:")
    print(f"  Shape: {rel_pos.shape}")
    print(f"  dog→person: dx={rel_pos[0, 1, 0]:.3f}, dy={rel_pos[0, 1, 1]:.3f}, iou={rel_pos[0, 1, 6]:.3f}")