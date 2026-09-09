from util.parser import get_parser_with_args
from util.metrics import FocalLoss, dice_loss,TverskyLoss
from util.lovasz_softmax import lovasz_softmax
from util.MultiTverskyLoss import MultiTverskyLoss




def hybrid_loss(predictions, target,device):
    "Calculating the loss"
    loss = 0

    # gamma=0, alpha=None --> CE
    focal = FocalLoss(gamma=0, alpha=0)
    # tv=TverskyLoss(alpha=0.3, beta=0.7)
  

    for prediction in predictions:

        bce = focal(prediction, target)
        dice = dice_loss(prediction, target,device)
       # dice = lovasz_softmax(prediction, target)
        # dice=tv(prediction, target,device)
        
        loss += bce + dice 

    return loss
    
    




##########################################33
from util.parser import get_parser_with_args
from util.metrics import FocalLoss, dice_loss,TverskyLoss
from util.lovasz_softmax import lovasz_softmax
from util.MultiTverskyLoss import MultiTverskyLoss
import torch  #new9.3
import torch.nn.functional as F  #new9.3



def hybrid_loss(predictions, target,device):
    "Calculating the loss"
    loss = 0

    # gamma=0, alpha=None --> CE
    focal = FocalLoss(gamma=0, alpha=0)
    # tv=TverskyLoss(alpha=0.3, beta=0.7)
  

    for prediction in predictions:

        bce = focal(prediction, target)
        dice = dice_loss(prediction, target,device)
       # dice = lovasz_softmax(prediction, target)
        # dice=tv(prediction, target,device)
        
        boundary = boundary_loss(prediction, target, device)   # 9.3新增
        
        #loss += bce + dice 9.3改动
        loss += bce + dice + 0.5 * boundary                     # λ 先取 0.5 试

    return loss

def boundary_loss(prediction, target, device, band=2):
    """边界加权CE：在GT边界带(内外各band像素)上加强监督，
    把向外扩张的预测边界拉回GT边界，改善 Precision 与 ASSD。
    prediction:[B,2,H,W] logits; target:[B,H,W] long(0/1)
    """
    gt = target.unsqueeze(1).float()                       # [B,1,H,W]
    k = band * 2 + 1
    eroded = -F.max_pool2d(-gt, kernel_size=k, stride=1, padding=band)   # 腐蚀:GT向内缩
    dilated = F.max_pool2d(gt, kernel_size=k, stride=1, padding=band)    # 膨胀:GT向外扩
    band_mask = (dilated - eroded).clamp(0, 1)             # 1=边界带像素
    prob = F.softmax(prediction, dim=1)
    prob_fg = prob[:, 1:2, :, :]
    eps = 1e-7
    ce = -(gt * torch.log(prob_fg + eps) + (1 - gt) * torch.log(1 - prob_fg + eps))
    loss = (ce * band_mask).sum() / (band_mask.sum() + 1e-7)
    return loss


