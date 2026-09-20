import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyAwareFeatureEnhancement(nn.Module):
    """
    Frequency-Aware Feature Enhancement Module (FAEM)

    功能：
    1. 将高层特征映射到频域
    2. 分离低频和高频信息
    3. 分别进行特征建模
    4. 自适应融合空间域、低频、高频特征
    5. 残差连接保持原始语义信息

    输入:
        x: [B, C, H, W]

    输出:
        out: [B, C, H, W]
    """

    def __init__(self, channels, reduction=16):
        super(FrequencyAwareFeatureEnhancement, self).__init__()

        self.channels = channels

        # =========================================================
        # 1. 空间域分支
        # =========================================================
        self.spatial_branch = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False
            ),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )

        # =========================================================
        # 2. 低频分支
        # =========================================================
        self.low_freq_branch = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False
            ),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )

        # =========================================================
        # 3. 高频分支
        # =========================================================
        self.high_freq_branch = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False
            ),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )

        # =========================================================
        # 4. 频率分支自适应权重
        #
        # 根据低频 / 高频特征自动学习：
        # low-frequency weight
        # high-frequency weight
        # =========================================================
        hidden_channels = max(channels // reduction, 16)

        self.frequency_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),

            nn.Conv2d(
                channels * 2,
                hidden_channels,
                kernel_size=1,
                bias=False
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                hidden_channels,
                channels * 2,
                kernel_size=1,
                bias=True
            ),

            nn.Sigmoid()
        )

        # =========================================================
        # 5. 空间 + 频域融合
        # =========================================================
        self.fusion = nn.Sequential(
            nn.Conv2d(
                channels * 3,
                channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )

        # =========================================================
        # 6. 可学习的频域增强比例
        #
        # 使用 sigmoid 保证权重稳定在 0~1
        # =========================================================
        self.low_scale = nn.Parameter(torch.tensor(0.0))
        self.high_scale = nn.Parameter(torch.tensor(0.0))
        
        # 新增
        self.res_scale = nn.Parameter(torch.tensor(0.0))

    def _frequency_decompose(self, x):
        """
        FFT频域分解

        将输入划分成：
        low-frequency
        high-frequency
        """

        B, C, H, W = x.shape

        # ---------------------------------------------------------
        # FFT
        # ---------------------------------------------------------
        fft = torch.fft.fft2(x, dim=(-2, -1))

        # ---------------------------------------------------------
        # 构造归一化频率坐标
        # ---------------------------------------------------------
        fy = torch.fft.fftfreq(
            H,
            device=x.device,
            dtype=x.dtype
        ).reshape(H, 1)

        fx = torch.fft.fftfreq(
            W,
            device=x.device,
            dtype=x.dtype
        ).reshape(1, W)

        radius = torch.sqrt(
            fx ** 2 + fy ** 2
        )

        # ---------------------------------------------------------
        # 低频区域
        #
        # 半径越小 -> 越接近低频
        # ---------------------------------------------------------
        cutoff = 0.25

        low_mask = (
            radius <= cutoff
        ).to(x.dtype)

        high_mask = 1.0 - low_mask

        low_mask = low_mask.unsqueeze(0).unsqueeze(0)
        high_mask = high_mask.unsqueeze(0).unsqueeze(0)

        # ---------------------------------------------------------
        # 频域分离
        # ---------------------------------------------------------
        low_fft = fft * low_mask
        high_fft = fft * high_mask

        # ---------------------------------------------------------
        # IFFT回到空间域
        # ---------------------------------------------------------
        low_freq = torch.fft.ifft2(
            low_fft,
            dim=(-2, -1)
        ).real

        high_freq = torch.fft.ifft2(
            high_fft,
            dim=(-2, -1)
        ).real

        return low_freq, high_freq

    def forward(self, x):

        identity = x

        # =========================================================
        # Step 1：空间域特征
        # =========================================================
        spatial_feat = self.spatial_branch(x)

        # =========================================================
        # Step 2：频域分解
        # =========================================================
        low_freq, high_freq = self._frequency_decompose(x)

        # =========================================================
        # Step 3：低频 / 高频特征提取
        # =========================================================
        low_feat = self.low_freq_branch(low_freq)

        high_feat = self.high_freq_branch(high_freq)

        # =========================================================
        # Step 4：学习低频 / 高频权重
        # =========================================================
        freq_cat = torch.cat(
            [low_feat, high_feat],
            dim=1
        )

        attention = self.frequency_attention(freq_cat)

        low_attention, high_attention = torch.chunk(
            attention,
            2,
            dim=1
        )

        # =========================================================
        # Step 5：可学习增强比例
        # =========================================================
        low_scale = torch.sigmoid(self.low_scale)
        high_scale = torch.sigmoid(self.high_scale)

        low_feat = (
            low_feat *
            low_attention *
            low_scale
        )

        high_feat = (
            high_feat *
            high_attention *
            high_scale
        )

        # =========================================================
        # Step 6：空间 + 低频 + 高频融合
        # =========================================================
        fusion_feat = torch.cat(
            [
                spatial_feat,
                low_feat,
                high_feat
            ],
            dim=1
        )

        fusion_feat = self.fusion(fusion_feat)

        # =========================================================
        # Step 7：残差连接
        # =========================================================
        #out = identity + fusion_feat
        out = identity + self.res_scale * fusion_feat

        return out