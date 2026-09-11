#!/usr/bin/env python3
"""
lavis_eva_vit.py — LAVIS eva_vit.py 精简内嵌版

从 Salesforce LAVIS 的 lavis/models/eva_vit.py 抽取出 BLIP-2 用的 EVA-ViT-g/14
最小可用子集。去除了 window attention / rel_pos_bias 等不相关分支,
只保留 BLIP-2 实际使用的 abs_pos_emb + qkv_bias 路径。

加载 eva_vit_g.pth 后 forward 返回 [B, 257, 1408] (含 CLS),
与 LAVIS 原版行为完全一致,也就是 EVCap 构建 ext_memory_lvis.pkl 时使用的编码器。

用法:
    from lavis_eva_vit import VisionTransformer, interpolate_pos_embed
    model = VisionTransformer(
        img_size=224, patch_size=14, use_mean_pooling=False,
        embed_dim=1408, depth=39, num_heads=16, mlp_ratio=4.3637,
        qkv_bias=True, drop_path_rate=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    state_dict = torch.load('eva_vit_g.pth', map_location='cpu')
    interpolate_pos_embed(model, state_dict)
    model.load_state_dict(state_dict, strict=False)
"""

import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── utils (避免依赖 timm, 自实现最小版本) ─────────────────────────
def _to_2tuple(x):
    if isinstance(x, (tuple, list)):
        return tuple(x)
    return (x, x)


def _drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


def _trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # 简化版 truncated normal (足够初始化用)
    with torch.no_grad():
        tensor.normal_(mean, std).clamp_(a, b)
    return tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return _drop_path(x, self.drop_prob, self.training)


# ── Mlp ──────────────────────────────────────────────────────────
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# ── Attention (EVA 风格: qkv 共享权重 + 独立 q_bias/v_bias) ───────
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rel_pos_bias=None):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((
                self.q_bias,
                torch.zeros_like(self.v_bias, requires_grad=False),
                self.v_bias,
            ))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ── Block ────────────────────────────────────────────────────────
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., init_values=None,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
                              attn_head_dim=attn_head_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)
        if init_values is not None and init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(dim))
            self.gamma_2 = nn.Parameter(init_values * torch.ones(dim))
        else:
            self.gamma_1 = None
            self.gamma_2 = None

    def forward(self, x, rel_pos_bias=None):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x), rel_pos_bias))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), rel_pos_bias))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


# ── PatchEmbed ──────────────────────────────────────────────────
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = _to_2tuple(img_size)
        patch_size = _to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


# ── VisionTransformer (EVA-ViT-g BLIP-2 配置) ────────────────────
class VisionTransformer(nn.Module):
    """EVA-ViT 视觉编码器 (LAVIS/BLIP-2 精简版)

    关键特性:
      - abs pos_embed (不用 rel_pos_bias)
      - qkv_bias=True (q_bias + v_bias, 没有 k_bias)
      - use_mean_pooling=False → self.norm = Identity
        → forward 返回未经 final norm 的特征, 交给外部 ln_vision 处理
      - forward 返回 [B, num_patches+1, embed_dim] = [B, 257, 1408]
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, init_values=None,
                 use_abs_pos_emb=True, use_mean_pooling=True,
                 use_checkpoint=False):
        super().__init__()
        self.image_size = img_size
        self.num_features = self.embed_dim = embed_dim
        self.use_checkpoint = use_checkpoint

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                      in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            self.pos_embed = None
        self.pos_drop = nn.Dropout(p=drop_rate)

        # BLIP-2 eva_vit_g 不用 rel_pos_bias
        self.rel_pos_bias = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop=drop_rate, attn_drop=attn_drop_rate,
                  drop_path=dpr[i], norm_layer=norm_layer,
                  init_values=init_values)
            for i in range(depth)
        ])

        # BLIP-2 用 use_mean_pooling=False, self.norm 是 Identity
        # 真正的 norm 在外面的 ln_vision (blip2 checkpoint 里)
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)

        if self.pos_embed is not None:
            _trunc_normal_(self.pos_embed, std=.02)
        _trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            _trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.patch_embed(x)
        B, N, _ = x.shape

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            if self.use_checkpoint:
                x = torch.utils.checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        # use_mean_pooling=False → self.norm=Identity, 这里返回未 norm 的特征
        # 实际的 ln_vision 在 blip2 checkpoint 里, 由调用方加
        return x

    def forward(self, x):
        return self.forward_features(x)


# ── interpolate_pos_embed (处理不同 img_size 时的 pos_embed 插值) ──
def interpolate_pos_embed(model, checkpoint_model):
    """如果 checkpoint 的 pos_embed 尺寸和 model 不一致, 做 bicubic 插值
    (BLIP-2 标准 224→224 时是 no-op)
    """
    if 'pos_embed' not in checkpoint_model:
        return
    pos_embed_checkpoint = checkpoint_model['pos_embed'].float()
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = model.patch_embed.num_patches
    num_extra_tokens = model.pos_embed.shape[-2] - num_patches

    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    new_size = int(num_patches ** 0.5)

    if orig_size != new_size:
        print(f"  [pos_embed interpolate] {orig_size}x{orig_size} → {new_size}x{new_size}")
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = F.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model['pos_embed'] = new_pos_embed


def create_eva_vit_g(img_size=224, drop_path_rate=0.0,
                     use_checkpoint=False, precision="fp16"):
    """LAVIS 风格的工厂函数, 和 LAVIS 的 create_eva_vit_g 同参"""
    model = VisionTransformer(
        img_size         = img_size,
        patch_size       = 14,
        use_mean_pooling = False,     # BLIP-2 不在这里做 norm
        embed_dim        = 1408,
        depth            = 39,
        num_heads        = 1408 // 88,  # = 16
        mlp_ratio        = 4.3637,
        qkv_bias         = True,
        drop_path_rate   = drop_path_rate,
        norm_layer       = partial(nn.LayerNorm, eps=1e-6),
        use_checkpoint   = use_checkpoint,
    )
    return model