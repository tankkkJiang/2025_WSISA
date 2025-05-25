#!/usr/bin/env python
# cluster_select_cnnsurv.py


import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from cnn_survival import CNNSurv, CoxPHLoss, c_index_torch

# ----------------------------------------
# 全局路径和设备配置
BASE       = os.path.abspath(os.path.dirname(__file__))
EXP_CSV    = os.path.join(BASE, "cluster_result", "patches_1000_cls10_expanded.csv")
PATCHES    = os.path.join(BASE, "data", "patches")
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# 超参数设置
EPOCHS, BATCH, LR, SEED = 20, 30, 5e-4, 1
torch.manual_seed(SEED)
np.random.seed(SEED)

# -------------------- 数据集定义 --------------------
class PatchDataset(Dataset):
    """
    只保留传入 DataFrame 中的行，用于 Patch 级别的训练/测试
    """
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        # 图像预处理：转 Tensor 并归一化到 [-1,1]
        self.tr = T.Compose([
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 按行读取图像路径、surv、status
        row = self.df.iloc[idx]
        img_path = os.path.join(BASE, row.patch_path)
        img = Image.open(img_path).convert("RGB")
        x = self.tr(img)
        t = torch.tensor(row.surv,   dtype=torch.float32)
        e = torch.tensor(row.status, dtype=torch.float32)
        return x, t, e

# -------------------- 训练函数 --------------------
def train_on_df(train_df):
    """
    在给定的 DataFrame 上训练 POI 网络，返回训练完成的模型
    """
    print(f"\n[TRAIN] 开始训练，训练集患者数量：{train_df.pid.nunique()}，Patch 数量：{len(train_df)}")
    loader = DataLoader(
        PatchDataset(train_df),
        batch_size=BATCH, shuffle=True, num_workers=4
    )
    # 构建模型、优化器、损失函数
    in_ch = next(iter(loader))[0].shape[1]
    model = CNNSurv(in_ch).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = CoxPHLoss()

    # 训练 EPOCHS 轮
    model.train()
    for ep in range(1, EPOCHS+1):
        epoch_loss = 0.0
        for x, t, e in loader:
            x, t, e = x.to(DEVICE), t.to(DEVICE), e.to(DEVICE)
            loss = loss_fn(model(x), t, e)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"[TRAIN] Epoch {ep}/{EPOCHS} 完成，平均 Loss = {epoch_loss/len(loader):.4f}")
    return model

# -------------------- 留一法主流程 --------------------
def loocv_and_aggregate():
    # 1) 读取扩展标签 CSV
    df = pd.read_csv(EXP_CSV)
    pids = df.pid.unique()

    all_risk = []  # 汇总所有 patch 的风险
    all_t    = []  # 汇总所有 patch 的生存时间
    all_e    = []  # 汇总所有 patch 的事件指示

    # 对每个患者留一法
    for pid in pids:
        print(f"\n[LOOCV] 当前留出患者: {pid}")
        train_df = df[df.pid != pid]
        test_df  = df[df.pid == pid]

        # 2) 训练模型
        model = train_on_df(train_df)

        # 3) 在留出患者上做预测
        print(f"[PREDICT] 在患者 {pid} 的 {len(test_df)} 个 patch 上预测风险")
        test_loader = DataLoader(
            PatchDataset(test_df), batch_size=BATCH,
            shuffle=False, num_workers=4
        )
        model.eval()
        with torch.no_grad():
            for x, t, e in test_loader:
                x, t, e = x.to(DEVICE), t.to(DEVICE), e.to(DEVICE)
                r = model(x).cpu()
                all_risk.append(r)
                all_t.append(t.cpu())
                all_e.append(e.cpu())

    # 4) 聚合预测并计算最终 C-index
    risks = torch.cat(all_risk).squeeze().numpy()
    times = torch.cat(all_t).squeeze().numpy()
    evs   = torch.cat(all_e).squeeze().numpy()
    cidx = c_index_torch(
        torch.from_numpy(risks),
        torch.from_numpy(times),
        torch.from_numpy(evs)
    )
    print(f"\n[RESULT] LOOCV 聚合后的 C-index = {cidx:.4f}")

if __name__ == "__main__":
    print("===== 开始 LOOCV 聚合 C-index 计算 =====")
    loocv_and_aggregate()
