import os, glob, random, re, zipfile
from PIL import Image
import numpy as np

SRC = "/media/ubuntu/Student/wt/EMGANet/Dataset/Dataset_B"
XLSX = os.path.join(SRC, "DatasetB.xlsx")
DST = "EMGANet/Dataset/Dataset_B"
SEED = 123

# ---------- 1. 零依赖解析 xlsx（标准库，不用 openpyxl）----------
def parse_xlsx(path):
    z = zipfile.ZipFile(path)
    ss = [re.sub(r'<[^>]+>', '', m) for m in
          re.findall(r'<si>(.*?)</si>', z.read('xl/sharedStrings.xml').decode('utf-8'), re.S)]
    sheet = z.read('xl/worksheets/sheet1.xml').decode('utf-8')
    label = {}
    for rowm in re.finditer(r'<row[^>]*r="(\d+)"[^>]*>(.*?)</row>', sheet, re.S):
        rn, body = int(rowm.group(1)), rowm.group(2)
        cells = {}
        for cm in re.finditer(r'<c r="([A-Z]+)\d+"(?:[^>]*t="(\w+)")?[^>]*>(?:<v>(.*?)</v>)?</c>', body, re.S):
            letter, t, v = cm.group(1), cm.group(2), cm.group(3)
            if v is None: continue
            cells[ord(letter)-65] = ss[int(v)] if t == 's' else v
        if rn == 1: header = cells
        elif 0 in cells and 1 in cells:
            label[str(cells[0]).zfill(6)] = (cells[1], cells[2] if 2 in cells else '')
    return label

labels = parse_xlsx(XLSX)          # {'000001': ('Benign','CYST'), ...}
assert len(labels) == 163, len(labels)
print('良恶性分布:', {t: sum(1 for _, (ty, _) in labels.items() if ty == t)
                     for t in ['Benign', 'Malignant']})   # 期望 Benign 110 / Malignant 53

# ---------- 2. 按良恶性分层随机划分 70/15/15 ----------
orig = sorted(glob.glob(os.path.join(SRC, "original", "*.png")))
gt   = sorted(glob.glob(os.path.join(SRC, "GT", "*.png")))
assert len(orig) == len(gt) == 163
random.seed(SEED)
train_ids, val_ids, test_ids = [], [], []
for typ in ['Benign', 'Malignant']:
    pool = [os.path.basename(p)[:6] for p in orig if labels[os.path.basename(p)[:6]][0] == typ]
    random.shuffle(pool)
    nt, nv = int(len(pool)*0.7), int(len(pool)*0.15)
    train_ids += pool[:nt]; val_ids += pool[nt:nt+nv]; test_ids += pool[nt+nv:]
split_of = {i: s for s, lst in [('train', train_ids), ('val', val_ids), ('test', test_ids)] for i in lst}
print('划分:', len(train_ids), len(val_ids), len(test_ids),
      '| test 内良恶性:', sum(1 for i in test_ids if labels[i][0]=='Benign'),
      sum(1 for i in test_ids if labels[i][0]=='Malignant'))

# ---------- 3. 重命名（编号+良恶性后缀，img/mask 同名）+ 转 BMP ----------
suffix = {'Benign': 'B', 'Malignant': 'M'}
for op, gp in zip(orig, gt):
    num = os.path.basename(op)[:6]
    name = "%s_%s.bmp" % (num, labels[num][0])   # 例：000001_B.bmp
    sp = split_of[num]
    for sub in ['img', 'mask']:
        os.makedirs(os.path.join(DST, sp, sub), exist_ok=True)
    Image.open(op).convert('L').save(os.path.join(DST, sp, 'img', name))
    m = np.array(Image.open(gp).convert('L'))
    m = np.where(m > 127, 255, 0).astype(np.uint8)        # GT 统一 0/255（关键）
    Image.fromarray(m).save(os.path.join(DST, sp, 'mask', name))

# ---------- 4. 落盘：划分名单 + 良恶性/病理映射表（阶段5用）----------
with open(os.path.join(DST, "split_ids.txt"), "w") as f:
    for s, lst in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        f.write("%s: %s\n" % (s, ",".join(sorted(lst))))
with open(os.path.join(DST, "labels.csv"), "w") as f:
    f.write("image,type,diagnosis,split\n")
    for num in sorted(labels):
        f.write("%s,%s,%s,%s\n" % (num, labels[num][0], labels[num][1], split_of[num]))
print("done. 文件示例: img=000001_B.bmp, mask=000001_B.bmp（同名配对）")
