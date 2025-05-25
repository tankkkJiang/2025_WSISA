#!/usr/bin/env python
# cluster_select_holdout.py

import os, random
import numpy as np, pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit

import torch, torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

from cnn_survival import CNNSurv, CoxPHLoss, c_index_torch

# ----------------------------------------------------------
BASE   = os.path.abspath(os.path.dirname(__file__))
PATCHES_ROOT      = os.path.join(BASE, "data", "patches")
PATIENT_LABEL_CSV = os.path.join(BASE, "data", "patients.csv")
EXP_LABEL_CSV     = os.path.join(
    BASE, "cluster_result", "patches_1000_cls10_expanded.csv"
)
LOG_DIR = os.path.join(BASE, "log", "wsisa_holdout_pt")
os.makedirs(LOG_DIR, exist_ok=True)

EPOCHS, BATCH, LR, SEED = 20, 30, 5e-4, 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
# ----------------------------------------------------------

class PatchDataset(Dataset):
    def __init__(self, df_idx):
        self.df = df_idx.reset_index(drop=True)
        self.tr = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(os.path.join(BASE, row.patch_path)).convert('RGB')
        x = self.tr(img)
        return (
            x,
            torch.tensor(row.surv, dtype=torch.float32),
            torch.tensor(row.status, dtype=torch.float32),
        )


def convert_index(pids, full_df):
    """根据 pid 列表映射出 patch 级索引列表"""
    return full_df.index[full_df.pid.isin(pids)].tolist()


def run_holdout(train_pids, val_pids, test_pids, expand_df):
    """训练／验证／测试一次 hold-out 划分"""
    # 1. 映射到 patch 行号
    train_idx = convert_index(train_pids, expand_df)
    val_idx   = convert_index(val_pids,   expand_df) if len(val_pids)>0 else []
    test_idx  = convert_index(test_pids,  expand_df)

    # 2. 保存索引 CSV
    for name, idx in [('train', train_idx), ('valid', val_idx), ('test', test_idx)]:
        path = os.path.join(LOG_DIR, f"{name}.csv")
        np.savetxt(path, idx, delimiter=',', header='index', comments='')

    # 3. DataLoader
    train_ld = DataLoader(PatchDataset(expand_df.loc[train_idx]),
                          batch_size=BATCH, shuffle=True, num_workers=4)
    val_ld   = DataLoader(PatchDataset(expand_df.loc[val_idx]),
                          batch_size=BATCH, shuffle=False, num_workers=4) if val_idx else None
    test_ld  = DataLoader(PatchDataset(expand_df.loc[test_idx]),
                          batch_size=BATCH, shuffle=False, num_workers=4)

    # 4. 模型、优化器、损失
    c, _, _ = next(iter(train_ld))[0].shape
    model    = CNNSurv(c).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = CoxPHLoss()

    # 5. 训练 + 验证
    best_val_c = 0.0
    for ep in range(1, EPOCHS+1):
        model.train()
        for x, t, e in train_ld:
            x, t, e = x.to(DEVICE), t.to(DEVICE), e.to(DEVICE)
            loss = criterion(model(x), t, e)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if val_ld:
            model.eval()
            all_r, all_t, all_e = [], [], []
            with torch.no_grad():
                for x, t, e in val_ld:
                    x, t, e = x.to(DEVICE), t.to(DEVICE), e.to(DEVICE)
                    all_r.append(model(x)); all_t.append(t); all_e.append(e)
            risk = torch.cat(all_r); tvec = torch.cat(all_t); evel = torch.cat(all_e)
            val_c = c_index_torch(risk, tvec, evel)
            best_val_c = max(best_val_c, val_c)
            print(f"Epoch {ep:02d}  val C-index = {val_c:.4f}")

    # 6. 测试
    model.eval()
    all_r, all_t, all_e = [], [], []
    with torch.no_grad():
        for x, t, e in test_ld:
            x, t, e = x.to(DEVICE), t.to(DEVICE), e.to(DEVICE)
            all_r.append(model(x)); all_t.append(t); all_e.append(e)
    risk = torch.cat(all_r); tvec = torch.cat(all_t); evel = torch.cat(all_e)
    test_c = c_index_torch(risk, tvec, evel)
    print(f"**TEST C-index = {test_c:.4f}**")

    return best_val_c, test_c


def main():
    # 读取扩展标签（patch 级）
    expand_df = pd.read_csv(EXP_LABEL_CSV)
    # 构建患者级标签表
    labels_df = expand_df[['pid','surv','status']].drop_duplicates().reset_index(drop=True)

    # 1. 固定一个患者做测试
    test_pid = labels_df.pid.iloc[0]   # 也可以改成任意 labels_df.pid.values[0]
    remaining = labels_df.pid[labels_df.pid != test_pid].values

    # 2. 从剩余患者中抽 1 名作验证，其余训练
    y_remain = labels_df.set_index('pid').loc[remaining, 'status'].values
    sss = StratifiedShuffleSplit(n_splits=1, test_size=1/len(remaining), random_state=SEED)
    tr_idx, va_idx = next(sss.split(np.zeros(len(remaining)), y_remain))
    train_pids = remaining[tr_idx]
    val_pids   = remaining[va_idx]

    print("Train PIDs:", train_pids)
    print("Valid PIDs:", val_pids)
    print("Test  PID:", test_pid)

    # 3. 运行一次 hold-out
    best_val_c, test_c = run_holdout(train_pids, val_pids, [test_pid], expand_df)

    print("Final best val C-index =", best_val_c)
    print("Final test C-index    =", test_c)


if __name__ == '__main__':
    main()