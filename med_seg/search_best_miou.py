import os
import sys
import random
import numpy as np
import pandas as pd
import torch

from glob import glob
from collections import OrderedDict
from torch.utils.data import DataLoader

# ============================================================
# 1. 路径
# ============================================================
PROJECT_ROOT = "/media/ubuntu/Student/wt/EMGANet"
MED_SEG = os.path.join(PROJECT_ROOT, "med_seg")

sys.path.insert(0, MED_SEG)

# ============================================================
# 2. 导入作者代码
# ============================================================
from dataset import CrackData
from util.transforms import train_transforms, test_transforms
from util.helpers import get_criterion
from util.common import ScaleInOutput
from models.seg_model import Seg_Detection

# ============================================================
# 3. 构造与作者完全一致的 opt
# ============================================================
class Opt:
    backbone = "swinv2_128"
    neck = "fpn+aspp+fuse+drop"
    head = "fcn"
    loss_function = "hybrid"
    pretrain = ""
    input_size = 256
    num_workers = 1
    #batch_size = 4
    batch_size = 4
    learning_rate = 0.003
    epochs = 1000


opt = Opt()

# ============================================================
# 4. 固定随机种子：与作者一致
# ============================================================
seed = 123

random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# ============================================================
# 5. CUDA
# ============================================================
assert torch.cuda.is_available(), "CUDA 不可用!"

device = torch.device("cuda:0")

print("=" * 70)
print("EMGANet One-Batch Training Test")
print("=" * 70)

print("PyTorch :", torch.__version__)
print("CUDA    :", torch.version.cuda)
print("GPU     :", torch.cuda.get_device_name(0))
print("Device  :", device)

# ============================================================
# 6. 数据集
# ============================================================
base = os.path.join(PROJECT_ROOT, "Dataset", "BUSI_WHU")

train_path = os.path.join(base, "train")

train_data = pd.DataFrame({
    "images": sorted(
        glob(os.path.join(train_path, "img", "*.bmp"))
    ),
    "masks": sorted(
        glob(os.path.join(train_path, "mask", "*.bmp"))
    )
})

print("\n" + "=" * 70)
print("Dataset")
print("=" * 70)

print("Train images:", len(train_data))
print("Train masks :", len(train_data))

assert len(train_data) == 647, \
    f"训练集数量异常: {len(train_data)}"

# ============================================================
# 7. Dataset：完全使用作者实现
# ============================================================
train_dataset = CrackData(
    df=train_data,
    transforms=train_transforms
)

train_loader = DataLoader(
    dataset=train_dataset,
    num_workers=opt.num_workers,
    batch_size=opt.batch_size,
    shuffle=True
)

# ============================================================
# 8. 取一个 batch
# ============================================================
batch_img, labels = next(iter(train_loader))

print("\n" + "=" * 70)
print("Raw DataLoader Batch")
print("=" * 70)

print("Images:")
print("  shape :", batch_img.shape)
print("  dtype :", batch_img.dtype)
print("  min   :", batch_img.min().item())
print("  max   :", batch_img.max().item())

print("Labels:")
print("  shape :", labels.shape)
print("  dtype :", labels.dtype)
print("  min   :", labels.min().item())
print("  max   :", labels.max().item())
print("  unique:", torch.unique(labels))

# ============================================================
# 9. 与作者训练代码一致
# ============================================================
batch_img = batch_img.float().to(device)
labels = labels.long().to(device)

print("\n" + "=" * 70)
print("After CUDA")
print("=" * 70)

print("Images:", batch_img.shape, batch_img.dtype, batch_img.device)
print("Labels:", labels.shape, labels.dtype, labels.device)

# ============================================================
# 10. ScaleInOutput：完全使用作者实现
# ============================================================
scale = ScaleInOutput(opt.input_size)

batch_img, batch_img2 = scale.scale_input(
    (batch_img, batch_img)
)

print("\n" + "=" * 70)
print("After ScaleInOutput")
print("=" * 70)

print("batch_img :", batch_img.shape)
print("batch_img2:", batch_img2.shape)

# ============================================================
# 11. 创建模型
# ============================================================
print("\n" + "=" * 70)
print("Building EMGANet")
print("=" * 70)

model = Seg_Detection(opt)

print("Model created successfully.")

# 作者代码中的初始化方式
model_dict = model.state_dict()

model.load_state_dict(
    OrderedDict(model_dict),
    strict=False
)

model = model.to(device)

print("Model moved to:", device)

# 参数量
total_params = sum(
    p.numel() for p in model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("Total parameters    :", total_params)
print("Trainable parameters:", trainable_params)

# ============================================================
# 12. Criterion：作者 Hybrid Loss
# ============================================================
criterion = get_criterion(opt)

print("\nCriterion:", criterion)

# ============================================================
# 13. Optimizer：完全按照作者
# ============================================================
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=opt.learning_rate,
    weight_decay=0.001
)

# ============================================================
# 14. Forward
# ============================================================
print("\n" + "=" * 70)
print("Forward")
print("=" * 70)

model.train()

torch.cuda.reset_peak_memory_stats()

optimizer.zero_grad()

cd_preds = model(batch_img)

print("Model forward completed.")

print("Prediction type:", type(cd_preds))

if isinstance(cd_preds, (tuple, list)):
    print("Number of predictions:", len(cd_preds))

    for i, pred in enumerate(cd_preds):
        print(
            f"Prediction[{i}]: "
            f"shape={pred.shape}, "
            f"dtype={pred.dtype}, "
            f"device={pred.device}, "
            f"min={pred.min().item():.6f}, "
            f"max={pred.max().item():.6f}"
        )
else:
    print(
        "Prediction:",
        cd_preds.shape,
        cd_preds.dtype,
        cd_preds.device
    )

# ============================================================
# 15. Scale Output
# ============================================================
cd_preds = scale.scale_output(cd_preds)

print("\n" + "=" * 70)
print("After Scale Output")
print("=" * 70)

if isinstance(cd_preds, (tuple, list)):
    print("Number of predictions:", len(cd_preds))

    for i, pred in enumerate(cd_preds):
        print(
            f"Prediction[{i}]: "
            f"shape={pred.shape}, "
            f"dtype={pred.dtype}, "
            f"device={pred.device}"
        )
else:
    print(
        "Prediction:",
        cd_preds.shape
    )

# ============================================================
# 16. Hybrid Loss
# ============================================================
print("\n" + "=" * 70)
print("Hybrid Loss")
print("=" * 70)

cd_loss = criterion(
    cd_preds,
    labels,
    device
)

print("Loss:", cd_loss.item())
print("Loss dtype:", cd_loss.dtype)
print("Loss device:", cd_loss.device)

assert torch.isfinite(cd_loss), \
    "Loss 出现 NaN 或 Inf!"

# ============================================================
# 17. Backward
# ============================================================
print("\n" + "=" * 70)
print("Backward")
print("=" * 70)

cd_loss.backward()

print("Backward completed successfully.")

# ============================================================
# 18. 检查梯度
# ============================================================
grad_count = 0
grad_nan_count = 0

for name, param in model.named_parameters():

    if param.grad is not None:

        grad_count += 1

        if not torch.isfinite(param.grad).all():
            grad_nan_count += 1

print("Parameters with gradients:", grad_count)
print("Parameters with NaN/Inf gradients:", grad_nan_count)

assert grad_count > 0, \
    "没有检测到任何梯度!"

assert grad_nan_count == 0, \
    "检测到 NaN/Inf 梯度!"

# ============================================================
# 19. Optimizer Step
# ============================================================
print("\n" + "=" * 70)
print("Optimizer Step")
print("=" * 70)

optimizer.step()

print("Optimizer step completed successfully.")

# ============================================================
# 20. GPU Memory
# ============================================================
allocated = torch.cuda.memory_allocated() / 1024**3
reserved = torch.cuda.memory_reserved() / 1024**3
peak = torch.cuda.max_memory_allocated() / 1024**3

print("\n" + "=" * 70)
print("GPU Memory")
print("=" * 70)

print(f"Current allocated : {allocated:.2f} GB")
print(f"Current reserved  : {reserved:.2f} GB")
print(f"Peak allocated    : {peak:.2f} GB")

# ============================================================
# 21. 最终结论
# ============================================================
print("\n" + "=" * 70)
print("✓ ONE-BATCH TEST PASSED")
print("=" * 70)

print("""
数据读取       ✓
DataLoader     ✓
CUDA           ✓
Model Forward  ✓
Scale Output   ✓
Hybrid Loss    ✓
Backward       ✓
Gradient       ✓
Optimizer      ✓

现在可以进入 3-epoch Debug Training。
""")
