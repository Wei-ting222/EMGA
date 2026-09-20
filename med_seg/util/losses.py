import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from util.parser import get_parser_with_args
from util.metrics import FocalLoss, dice_loss,TverskyLoss
from util.lovasz_softmax import lovasz_softmax
from util.MultiTverskyLoss import MultiTverskyLoss


def _signed_distance(mask):
    """单张 0/1 掩膜 -> 符号距离图 phi。
    病灶外部：到病灶边界的距离（正）；病灶内部：到边界距离取负（Kervadec 约定，内部 -(d-1)）。
    """
    fg = mask > 0.5
    if not fg.any():
        # 没有前景时返回全 0，避免空掩膜产生 NaN（BUSI 良性/恶性样本一般都有病灶，这里只是兜底）
        return np.zeros_like(mask, dtype=np.float32)
    dist_out = distance_transform_edt(~fg)     # 背景各点到前景的距离
    dist_in = distance_transform_edt(fg)       # 前景各点到背景的距离
    phi = np.where(fg, -(dist_in - 1.0), dist_out).astype(np.float32)
    return phi


def boundary_loss(logits, target, normalize=True):
    """Kervadec Boundary Loss（MIDL 2019，github.com/LIVIAETS/boundary-loss）。
    在线由「当前 batch 增强后的 GT」实时计算符号距离图，因此与随机翻转/旋转等
    数据增强天然对齐，不需要改 dataset/dataloader，也不用离线存距离图。
    Args:
        logits: [B, 2, H, W] 原始网络输出（未 softmax）。
        target: [B, H, W]，取值 0/1（ToTensor 已把 0/255 掩膜归一到 0/1）。
        normalize: 是否按病灶平均深度归一化，使损失量级与 BCE/Dice 相当，
                   若改成 False（完全忠于官方未归一化版本），权重需降到 0.005~0.01。
    """
    prob_fg = F.softmax(logits, dim=1)[:, 1]          # [B,H,W] 前景概率
    masks = target.detach().cpu().numpy()
    phis = []
    for b in range(masks.shape[0]):
        phi = _signed_distance((masks[b] > 0.5).astype(np.uint8))
        if normalize:
            inside = phi[phi < 0]
            denom = float(np.abs(inside).mean()) if inside.size > 0 else 1.0
            if denom > 1e-6:
                phi = phi / denom
        phis.append(phi)
    phi_t = torch.from_numpy(np.stack(phis)).float().to(logits.device)  # [B,H,W]
    # 前景概率落在病灶内(phi<0)给负奖励、错误扩张到背景(phi>0)给惩罚，从而把边界拉回 GT
    return (prob_fg * phi_t).mean()


def hybrid_loss(predictions, target, device, bd_lambda=0.0):
    "Calculating the loss"
    loss = 0
    # gamma=0, alpha=None --> CE
    focal = FocalLoss(gamma=0, alpha=0)
    # tv=TverskyLoss(alpha=0.3, beta=0.7)
    for prediction in predictions:
        bce = focal(prediction, target)
        dice = dice_loss(prediction, target, device)
        # dice = lovasz_softmax(prediction, target)
        # dice=tv(prediction, target,device)
        bd = boundary_loss(prediction, target)
        loss += bce + dice + bd_lambda * bd
    return loss