import datetime
import torch
from sklearn.metrics import precision_recall_fscore_support as prfs
from util.parser import get_parser_with_args
from util.helpers import (initialize_metrics, get_mean_metrics,
                          set_metrics)
import os
import logging
import json
import pandas as pd
from util.AverageMeter import AverageMeter, RunningMetrics
from tqdm import tqdm
import random
import numpy as np
import ml_collections
from torch.utils.data import DataLoader
from models.block.Drop import dropblock_step
from util.common import check_dirs, gpu_info, SaveResult, CosOneCycle, ScaleInOutput
import argparse
from models.seg_model import Seg_Detection
from util.transforms import train_transforms,test_transforms
from glob import glob
from collections import OrderedDict
# 退火版混合损失：BCE + Dice + bd_lambda * BoundaryLoss
from util.losses import hybrid_loss
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser('Seg Detection train')
parser.add_argument("--backbone", type=str, default="swinv2_128")
parser.add_argument("--neck", type=str, default="fpn+aspp+fuse+drop")
parser.add_argument("--head", type=str, default="fcn")
parser.add_argument("--loss_function", type=str, default="hybrid")
parser.add_argument("--pretrain", type=str, default='')
parser.add_argument("--input_size", type=int, default=256)
parser.add_argument("--num_workers", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--learning_rate", type=float, default=0.003)   # 必须 float
parser.add_argument("--epochs", type=int, default=1000)             # 与 WHU 统一 1000

opt = parser.parse_args()
print(torch.cuda.is_available())
device = torch.device("cuda:0")
logging.info('GPU AVAILABLE? ' + str(torch.cuda.is_available()))

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(seed=123)

from dataset import CrackData
print('===> Loading datasets')
# ===== 唯一与 WHU 不同的地方：数据路径 BUSI（452/96/99）=====
train_path = "EMGANet/Dataset/BUSI/train"
val_path = "EMGANet/Dataset/BUSI/val"
test_path =  "EMGANet/Dataset/BUSI/test"

train_data = pd.DataFrame({'images': sorted(glob(os.path.join(train_path, "img") + "/*.bmp")),
              'masks': sorted(glob(os.path.join(train_path, "mask") + "/*.bmp"))})
val_data = pd.DataFrame({'images': sorted(glob(os.path.join(val_path, "img") + "/*.bmp")),
              'masks': sorted(glob(os.path.join(val_path, "mask") + "/*.bmp"))})
test_data = pd.DataFrame({'images': sorted(glob(os.path.join(test_path, "img") + "/*.bmp")),
              'masks': sorted(glob(os.path.join(test_path, "mask") + "/*.bmp"))})

train_dataset = CrackData(df=train_data, transforms=train_transforms)
val_dataset = CrackData(df=val_data, transforms=test_transforms)
test_dataset = CrackData(df=test_data, transforms=test_transforms)
print(len(train_dataset), len(val_dataset), len(test_dataset))

train_loader = DataLoader(dataset=train_dataset, num_workers=opt.num_workers, batch_size=opt.batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, num_workers=opt.num_workers, batch_size=1, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, num_workers=opt.num_workers, batch_size=1, shuffle=False)
print(len(train_loader))

print('===> Building model')
save_path = check_dirs()
save_result = SaveResult(result_save_path=save_path + "/save_results.txt")
save_result.prepare()

logging.info('LOADING Model')
model = Seg_Detection(opt)
model_dict = model.state_dict()
model.load_state_dict(OrderedDict(model_dict), strict=False)
model = model.to(device)
print("load weight~~~~~~~~~~~~~~~~~~~~~~~~~")

optimizer = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate, weight_decay=0.001)
scheduler = CosOneCycle(optimizer, max_lr=opt.learning_rate, epochs=opt.epochs, up_rate=0)

best_metrics = {'precision_1': -1, 'recall_1': -1, 'F1_1': -1, 'Overall_Acc': -1, 'Mean_IoU': -1}
logging.info('STARTING training')
total_step = -1
print('---------- Networks initialized -------------')
scale = ScaleInOutput(opt.input_size)

for epoch in range(opt.epochs):
    train_metrics = initialize_metrics()
    val_metrics = initialize_metrics()

    # ===== Boundary Loss 退火：前50轮 0（纯BCE+Dice预热），第50轮起 0.5 =====
    #bd_lambda = 0.0 if epoch < 50 else 0.25
    bd_lambda = 0.0   # C组：恒0，只测CARAFE，等价纯BCE+Dice
    print(f'[TRAIN INFO] Epoch:{epoch}, boundary_loss lambda = {bd_lambda}')

    model.train()
    logging.info('SET model mode to train!')
    batch_iter = 0
    tbar = tqdm(train_loader)
    total_train_loss = 0
    train_running_metrics = RunningMetrics(2)
    for batch_img, labels in tbar:
        tbar.set_description("epoch {} info ".format(epoch) + str(batch_iter) + " - " + str(batch_iter+opt.batch_size))
        batch_iter += opt.batch_size
        total_step += 1
        batch_img = batch_img.float().to(device)
        labels = labels.long().to(device)
        optimizer.zero_grad()
        batch_img, batch_img2 = scale.scale_input((batch_img, batch_img))
        cd_preds = model(batch_img)
        cd_preds = scale.scale_output(cd_preds)

        cd_loss = hybrid_loss(cd_preds, labels, device, bd_lambda=bd_lambda)
        loss = cd_loss
        loss.backward()
        optimizer.step()

        cd_preds = cd_preds[-1]
        _, cd_preds = torch.max(cd_preds, 1)
        train_running_metrics.update(labels.data.cpu().numpy(), cd_preds.data.cpu().numpy())
        total_train_loss += cd_loss.item()
        del batch_img, labels

    train_avg_loss = total_train_loss / len(train_loader)
    scheduler.step()
    dropblock_step(model)
    mean_train_metrics = train_running_metrics.get_scores()
    logging.info("EPOCH {} TRAIN METRICS".format(epoch) + str(mean_train_metrics))

    model.eval()
    total_val_loss = 0
    val_running_metrics = RunningMetrics(2)
    with torch.no_grad():
        for batch_img, labels in val_loader:
            batch_img = batch_img.float().to(device)
            labels = labels.long().to(device)
            batch_img, batch_img2 = scale.scale_input((batch_img, batch_img))
            cd_preds = model(batch_img)
            cd_preds = scale.scale_output(cd_preds)

            cd_loss = hybrid_loss(cd_preds, labels, device, bd_lambda=bd_lambda)

            cd_preds = cd_preds[-1]
            _, cd_preds = torch.max(cd_preds, 1)
            val_running_metrics.update(labels.data.cpu().numpy(), cd_preds.data.cpu().numpy())
            total_val_loss += cd_loss.item()
            del batch_img, labels

        val_avg_loss = total_val_loss / len(val_loader)
        mean_val_metrics = val_running_metrics.get_scores()
        logging.info("EPOCH {} VALIDATION METRICS".format(epoch) + str(mean_val_metrics))

        if mean_val_metrics['F1_1'] > best_metrics['F1_1']:
            print(f"[SAVE] epoch {epoch}: val F1_1={mean_val_metrics['F1_1']:.4f} val mIoU={mean_val_metrics['Mean_IoU']:.4f}")
            logging.info('updata the model')
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            with open(save_path+'/metadata_epoch_' + str(epoch) + '.json', 'w') as fout:
                torch.save(model, save_path+'/checkpoint_epoch_'+str(epoch)+'.pt')
            best_metrics = mean_val_metrics
            save_result.show(p=mean_train_metrics['precision_1'], r=mean_train_metrics['recall_1'],
                             f1=mean_train_metrics['F1_1'], miou=mean_train_metrics['Mean_IoU'],
                             oa=mean_train_metrics['Overall_Acc'], refer_metric=mean_train_metrics['Mean_IoU'],
                             best_metric=mean_val_metrics['Mean_IoU'],
                             train_avg_loss=train_avg_loss, val_avg_loss=val_avg_loss,
                             lr=opt.learning_rate, epoch=epoch)

        print('An epoch finished.')

print('Done!')
