import os, random
from glob import glob
from PIL import Image
import numpy as np

random.seed(123)

src = "/media/ubuntu/Student/wt/EMGANet/Dataset/Dataset_BUSI_with_GT"
dst_root = "/media/ubuntu/Student/wt/EMGANet/Dataset/BUSI"

# 1) 收集 benign + malignant 的所有图像，及每张图对应的【所有】掩码（排除 normal）
pairs = []  # (img_path, [mask_path, ...])
for cls in ["benign", "malignant"]:
    cls_dir = os.path.join(src, cls)
    images = [f for f in sorted(os.listdir(cls_dir))
              if f.endswith(".png") and "_mask" not in f]
    for f in images:
        base = f[:-4]  # 去掉 ".png"
        # 匹配 base_mask.png / base_mask_1.png / base_mask_2.png ...
        masks = sorted(g for g in os.listdir(cls_dir)
                       if g.startswith(base + "_mask") and g.endswith(".png"))
        if not masks:
            print("WARN no mask:", f)
            continue
        pairs.append((os.path.join(cls_dir, f),
                      [os.path.join(cls_dir, g) for g in masks]))

print("total images:", len(pairs))   # 应为 647

# 2) 按类别分层 70/15/15（保证良性/恶性在各子集比例一致）
def split_by_class(lst):
    random.shuffle(lst)
    n = len(lst)
    n_tr, n_va = int(n * 0.7), int(n * 0.15)
    return lst[:n_tr], lst[n_tr:n_tr + n_va], lst[n_tr + n_va:]

b_tr, b_va, b_te = split_by_class([p for p in pairs if "benign" in p[0]])
m_tr, m_va, m_te = split_by_class([p for p in pairs if "malignant" in p[0]])

splits = {"train": b_tr + m_tr, "val": b_va + m_va, "test": b_te + m_te}
for k in splits:
    random.shuffle(splits[k])

# 3) 合并多掩码（像素取并集），转 bmp 写入
for split_name, items in splits.items():
    img_dir = os.path.join(dst_root, split_name, "img")
    mask_dir = os.path.join(dst_root, split_name, "mask")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    for i, (img_p, mask_list) in enumerate(items):
        cls = "benign" if "benign" in img_p else "malignant"
        name = f"{cls}_{i:04d}.bmp"              # 例：benign_0000.bmp
        mask_name = f"{cls}_{i:04d}_mask.bmp"    # 例：benign_0000_mask.bmp
        Image.open(img_p).convert("RGB").save(os.path.join(img_dir, name))
        merged = None
        for mp in mask_list:
            m = np.array(Image.open(mp).convert("L"))
            merged = m if merged is None else np.maximum(merged, m)
        Image.fromarray(merged.astype("uint8")).save(os.path.join(mask_dir, mask_name))
        if len(mask_list) > 1:
            print("merged", len(mask_list), "masks for", img_p)
    

# 4) 校验
for split_name in ["train", "val", "test"]:
    ni = len(glob(os.path.join(dst_root, split_name, "img", "*.bmp")))
    nm = len(glob(os.path.join(dst_root, split_name, "mask", "*.bmp")))
    print(split_name, "img:", ni, "mask:", nm, "OK" if ni == nm else "MISMATCH")
