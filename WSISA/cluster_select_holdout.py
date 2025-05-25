#!/usr/bin/env python
# cluster_select_holdout.py
"""
单一 Hold-out 划分脚本，针对 7 个患者：
  1. 随机（或固定）选 1 个患者做测试
  2. 剩余 6 个里随机选 1 个做验证
  3. 剩余 5 个做训练
  4. 在 patch 级别上映射并训练/验证/测试
"""

import os, random
import numpy as np, pandas as pd
from PIL import Image
import torch
import torchvision.transforms as T
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
    """一次 Hold-out 划分：训练／验证／测试"""
    # 1. patch 级映射
    train_idx = convert_index(train_pids, expand_df)
    val_idx   = convert_index(val_pids,   expand_df) if len(val_pids)>0 else []
    test_idx  = convert_index(test_pids,  expand_df)

    # 2. 保存索引
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

    # 4. 模型 & 优化器
    in_ch = next(iter(train_ld))[0].shape[1]
    model    = CNNSurv(in_ch).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = CoxPHLoss()

    # 5. 训练 + 验证
    best_val_c = 0.0
    for ep in range(1, EPOCHS+1):
        model.train()
        for x, t, e in train_ld:
            x, t, e = x.to(DEVICE), t.to(DEVICE), e.to(DEVICE)
            loss = criterion(model(x), t, e)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

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
    # 读取 patch 级标签
    expand_df = pd.read_csv(EXP_LABEL_CSV)
    # 构建患者级标签
    labels_df = expand_df[['pid','surv','status']].drop_duplicates().reset_index(drop=True)
    all_pids  = labels_df.pid.values

    # 1) 随机（或固定）选一个做测试
    random.seed(SEED)
    test_pid = random.choice(all_pids)
    remaining = [pid for pid in all_pids if pid != test_pid]

    # 2) 在剩余里随机选一个做验证
    val_pid = random.choice(remaining)
    train_pids = [pid for pid in remaining if pid != val_pid]

    print("Train PIDs:", train_pids)
    print("Valid PID: ", val_pid)
    print("Test  PID: ", test_pid)

    # 3) 一次 Hold-out 训练/评估
    best_val_c, test_c = run_holdout(train_pids, [val_pid], [test_pid], expand_df)

    print("Final best val C-index =", best_val_c)
    print("Final test C-index    =", test_c)

if __name__ == '__main__':
    main()