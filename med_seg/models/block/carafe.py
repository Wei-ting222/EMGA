# -*- coding: utf-8 -*-
"""
CARAFE: Content-Aware ReAssembly of FEatures (Wang et al., ICCV 2019 Oral)
纯 PyTorch 实现（无需 CUDA 编译），改写自官方开源实现 leftthomas/CARAFE。

作用：用「特征内容」预测上采样卷积核，替代 nn.Upsample / F.interpolate 的
固定双线性上采样，让病灶边缘在放大时按内容重组，从而降低 ASSD、提升边界精度。

特性：输入/输出通道数不变，仅把空间分辨率放大 up_factor 倍，可直接 drop-in
替换解码器里的 nn.Upsample(scale_factor=2)。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CARAFE(nn.Module):
    def __init__(self, in_channels, kernel_size=5, up_factor=2, compress_ratio=4):
        super(CARAFE, self).__init__()
        self.up_factor = up_factor
        self.kernel_size = kernel_size

        mid_c = max(in_channels // compress_ratio, 1)
        # 1) 通道压缩，降低内容编码的计算量
        self.compress = nn.Conv2d(in_channels, mid_c, kernel_size=1)
        # 2) 内容编码器：为每个位置生成 up^2 * k^2 个重组权重
        self.encoder = nn.Conv2d(mid_c,
                                 up_factor ** 2 * kernel_size ** 2,
                                 kernel_size=kernel_size,
                                 padding=kernel_size // 2)
        # PixelShuffle 把通道维的 up^2 挪到空间维
        self.pix_shuffle = nn.PixelShuffle(up_factor)
        # 3) 取出每个输出位置对应的 k*k 邻域
        self.unfold = nn.Unfold(kernel_size=kernel_size,
                                dilation=up_factor,
                                padding=(kernel_size // 2) * up_factor)

    def forward(self, x):
        B, C, H, W = x.shape
        up = self.up_factor
        k = self.kernel_size

        # ---- 内容相关的重组权重：B, k*k, upH, upW，k*k 个权重 softmax 归一 ----
        w = self.compress(x)
        w = self.encoder(w)
        w = self.pix_shuffle(w)
        w = F.softmax(w, dim=1)

        # ---- 先最近邻放大，再 unfold 取邻域 ----
        x_up = F.interpolate(x, scale_factor=up, mode='nearest')
        x_up = self.unfold(x_up)                                   # B, C*k*k, upH*upW
        x_up = x_up.view(B, C, k * k, H * up, W * up)

        # ---- 按内容权重加权重组 ----
        w = w.unsqueeze(1)                                         # B, 1, k*k, upH, upW
        out = (x_up * w).sum(dim=2)                                # B, C, upH, upW
        return out
