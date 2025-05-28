#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WSISA/cluster_select_deepconvsurv.py

功能：
  • 每个簇各自训练 DeepConvSurv，计算测试集 C‑index
  • 训练集 / 测试集 = (所有患者‑5) / 5，保证两类 status 均包含
  • 仅保留 C‑index ≥ 0.5 的“有效簇”，写入 selected_clusters.txt
  • 关键优化：用 Dataset + DataLoader 按批加载 patch，避免内存爆掉
"""
import random, warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch, torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from networks import DeepConvSurv, NegativeLogLikelihood, c_index_torch

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------- 配置 ----------------
SEED = 1
random.seed(SEED);  torch.manual_seed(SEED)

ROOT       = Path(__file__).resolve().parent
PATCH_CSV  = ROOT / "cluster_result" / "patches_1000_cls10_expanded.csv"
MODEL_DIR  = ROOT / "log" / "wsisa_patch10" / "convimgmodel"
SEL_FILE   = ROOT / "log" / "selected_clusters.txt"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS, BATCH_SIZE, LR = 3, 16, 1e-4
NUM_WORKERS            = 4        # 调小可省显存 / CPU 内存
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
C_THRESH = 0.5                    # 有效簇阈值

MEAN = [0.6964, 0.5905, 0.6692]
STD  = [0.2559, 0.2943, 0.2462]
TRANSF = T.Compose([T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])

# ---------------- Dataset ----------------
class PatchDataset(Dataset):
    """按需读取 patch，返回 (tensor, surv, status)"""
    def __init__(self, rows: pd.DataFrame, root: Path):
        self.paths   = [root / p for p in rows["patch_path"].tolist()]
        self.surv    = rows["surv"].values.astype("float32")
        self.status  = rows["status"].values.astype("float32")

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        pth = self.paths[idx]
        try:
            img = Image.open(pth).convert("RGB")
            x   = TRANSF(img)
        except Exception as e:
            # 读错图片时返回全 0，占位同时打印一次 warning
            if random.random() < 1e-3:
                print(f"[Warn] 打开图像失败 {pth.name}: {e}")
            x = torch.zeros(3, 512, 512)
        return x, self.surv[idx], self.status[idx]

def make_loader(rows: pd.DataFrame, shuffle=False) -> DataLoader:
    ds = PatchDataset(rows, ROOT)
    return DataLoader(ds,
                      batch_size=BATCH_SIZE,
                      shuffle=shuffle,
                      num_workers=NUM_WORKERS,
                      pin_memory=torch.cuda.is_available(),
                      drop_last=False)

# ---------------- 读取 & 基本信息 ----------------
df = pd.read_csv(PATCH_CSV)

print("\n>>> cluster 分布:")
for cid, g in df.groupby("cluster"):
    print(f"  cluster {cid:2d}: {len(g):6d} patches")

pids = df["pid"].unique().tolist()
print(f"\n>>> 数据集共有 {len(pids)} 位病人\n")

pid_status = df.groupby("pid")["status"].apply(lambda x: sorted(x.unique())).to_dict()
print(">>> 各患者生存状态:")
for pid, sts in pid_status.items():
    print(f"  {pid}: {sts}")

# ---------------- 挑选 5 位做测试 ----------------
test_pids = None
for combo in combinations(pids, 5):
    sts = {s for pid in combo for s in pid_status[pid]}
    if sts == {0, 1}:
        test_pids = list(combo); break
if test_pids is None:
    raise RuntimeError("找不到同时包含存活和死亡的 5 位患者组合")

train_val_pids = [pid for pid in pids if pid not in test_pids]
print(f"\n>>> 测试集患者（5 位）：{test_pids}")
print(f">>> 训练/验证集患者（{len(train_val_pids)} 位）：{train_val_pids}\n")

# ---------------- per‑cluster 训练 + 推理 ----------------
clusters   = sorted(df["cluster"].unique())
fold_cidx  = {}
print(f"========== Fold 1 / 1 (train={len(train_val_pids)}, test=5) ==========")

for c in tqdm(clusters, desc="Clusters"):
    df_c  = df[df.cluster == c]
    df_tr = df_c[df_c.pid.isin(train_val_pids)]
    df_te = df_c[df_c.pid.isin(test_pids)]

    if len(df_tr) < 5:
        print(f"[Skip] cluster {c}: 训练样本 <5")
        fold_cidx[c] = 0.0;  continue

    # DataLoader
    train_loader = make_loader(df_tr, shuffle=True)
    test_loader  = make_loader(df_te, shuffle=False)

    # ----------- 训练 -----------
    model = DeepConvSurv(get_features=False).to(DEVICE)
    crit  = NegativeLogLikelihood()
    opt   = optim.Adam(model.parameters(), lr=LR)

    for _ in range(EPOCHS):
        model.train()
        for xb, tb, eb in train_loader:
            xb, tb, eb = xb.to(DEVICE), tb.to(DEVICE), eb.to(DEVICE)
            loss = crit(model(xb), tb, eb)
            opt.zero_grad();  loss.backward();  opt.step()

    # ----------- 测试 -----------
    model.eval();  preds = [];  T_list = [];  E_list = []
    with torch.no_grad():
        for xb, tb, eb in test_loader:
            preds.append(model(xb.to(DEVICE)).cpu())
            T_list.append(tb);  E_list.append(eb)
    if not preds:                     # 测试集中这个簇可能没有 patch
        print(f"[Warn] cluster {c}: 测试为空，C‑index=0")
        c_idx = 0.0
    else:
        y_pred = torch.cat(preds, dim=0)
        Tte    = torch.cat(T_list);  Ete = torch.cat(E_list)
        try:
            c_idx = c_index_torch(y_pred, Tte, Ete)
        except ZeroDivisionError:
            c_idx = 0.0

    fold_cidx[c] = c_idx
    print(f"Cluster {c:2d}  →  C-index = {c_idx:.4f}")

    # 保存模型
    torch.save({"model": model.state_dict()},
               MODEL_DIR / f"convimgmodel_cluster{c}_fold1.pth")

# ---------------- 筛选 & 保存 ----------------
selected = [c for c, v in fold_cidx.items() if v >= C_THRESH]
print("\n>>> C_THRESH =", C_THRESH)
print(">>> 满足阈值的簇：", selected)

with open(SEL_FILE, "w") as f:
    f.write(",".join(map(str, selected)))
print(f"[Saved] selected clusters → {SEL_FILE}")