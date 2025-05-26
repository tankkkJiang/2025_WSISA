#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WSISA/cluster_select_deepconvsurv.py

目的：在只有 7 位病人的数据集上也能完整跑一遍
      * 每个簇各自训练 1 个 DeepConvSurv
      * 直接用所有 patch 作为 train/valid/test（不关心数据泄漏）
      * 打印训练过程和 C-index（若无可比较样本对则记 0.0）
"""
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.optim as optim
import torchvision.transforms as T
from tqdm import tqdm

from networks import DeepConvSurv, NegativeLogLikelihood, c_index_torch

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------- 超参与路径 ----------------
SEED       = 1
random.seed(SEED)
torch.manual_seed(SEED)

ROOT       = Path(__file__).resolve().parent
PATCH_CSV  = ROOT / "cluster_result" / "patches_1000_cls10_expanded.csv"
MODEL_DIR  = ROOT / "log" / "wsisa_patch10" / "convimgmodel"
SEL_FILE   = ROOT / "log" / "selected_clusters.txt"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS, BATCH_SIZE, LR = 3, 64, 1e-4   # 减少 epoch，加快演示
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
C_THRESH = 0.0                         # 只要能算出 C-index 就保留

MEAN=[0.6964,0.5905,0.6692]
STD =[0.2559,0.2943,0.2462]
TRANSF = T.Compose([T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])

# ---------------- 读取数据 ----------------
df = pd.read_csv(PATCH_CSV)

print("\n>>> cluster 分布:")
for cid, g in df.groupby("cluster"):
    print(f"  cluster {cid:2d}: {len(g):6d} patches")

unique_pids = df["pid"].unique().tolist()
print(f"\n>>> 数据集共有 {len(unique_pids)} 位病人，全部参与训练/验证/测试\n")

all_idx = df.index.tolist()  # 同时作为 train/valid/test 的索引

def make_dataset(rows):
    xs, ts, es = [], [], []
    for _, r in tqdm(rows.iterrows(), total=len(rows), desc="  load patches", leave=False):
        try:
            img = Image.open(ROOT / r["patch_path"])
        except Exception as e:
            warnings.warn(f"[Warn] 无法打开图像 {r['patch_path']}: {e}", UserWarning)
            continue
        xs.append(TRANSF(img))
        ts.append(float(r["surv"]))
        es.append(float(r["status"]))
    if len(xs) == 0:
        warnings.warn(f"[Warn] make_dataset: rows={len(rows)} → 无有效 patch，返回空张量", UserWarning)
        return torch.Tensor(), torch.Tensor(), torch.Tensor()
    return torch.stack(xs), torch.tensor(ts), torch.tensor(es)

clusters = sorted(df["cluster"].unique())
fold_cidx = {c: [] for c in clusters}  # 只有一个 fold，但保持 dict 结构

# ---------------- 单折训练 ----------------
print("========== Fold 1 / 1  (all data) ==========")
for c in tqdm(clusters, desc="Clusters"):
    rows_c = df.loc[all_idx].query("cluster == @c")
    if len(rows_c) < 5:  # 样本太少
        print(f"[Skip] cluster {c} 样本 <5")
        fold_cidx[c].append(0.0)
        continue

    X, Tsurv, Estatus = make_dataset(rows_c)
    if X.numel() == 0:
        # 如果 make_dataset 返回空张量，跳过
        fold_cidx[c].append(0.0)
        print(f"[Skip] cluster {c} 无有效数据，已跳过")
        continue

    model = DeepConvSurv(get_features=False).to(DEVICE)
    crit  = NegativeLogLikelihood()
    opt   = optim.Adam(model.parameters(), lr=LR)

    for ep in tqdm(range(1, EPOCHS+1), desc=f"  cluster {c} epochs", leave=False):
        model.train()
        perm = torch.randperm(X.size(0))
        for i in tqdm(range(0, len(perm), BATCH_SIZE), desc="   batches", leave=False):
            idx = perm[i : i+BATCH_SIZE]
            x = X[idx].to(DEVICE)
            t = Tsurv[idx].to(DEVICE)
            e = Estatus[idx].to(DEVICE)
            loss = crit(model(x), t, e)
            opt.zero_grad()
            loss.backward()
            opt.step()

    # 计算 C-index（train==test）
    model.eval()
    with torch.no_grad():
        try:
            c_idx = c_index_torch(model(X.to(DEVICE)), Tsurv, Estatus)
        except ZeroDivisionError:
            c_idx = 0.0
            print(f"[Warn] cluster {c}: 无可比较样本对，C-index 置 0")
    fold_cidx[c].append(c_idx)

    print(f"Cluster {c:2d}  训练完成  →  C-index={c_idx:.4f}")
    # 保存权重
    torch.save({"model": model.state_dict()},
               MODEL_DIR / f"convimgmodel_cluster{c}_fold1.pth")

# ---------------- 选簇并保存列表 ----------------
selected = [c for c, v in fold_cidx.items() if np.mean(v) >= C_THRESH]
print("\n>>> 满足阈值的簇：", selected)
with open(SEL_FILE, "w") as f:
    f.write(",".join(map(str, selected)))
print(f"[Saved] selected cluster list → {SEL_FILE}")