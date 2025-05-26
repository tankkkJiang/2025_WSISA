#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WSISA/cluster_select_deepconvsurv.py  (简化版，仅为跑通 7 位病人数据)

- 不做 LOPO，只跑 1 折
- 7 位病人全部进训练集，再随机选 1 位 status=1 的人作验证
- 捕获 lifelines ZeroDivisionError，避免 "No admissable pairs" 崩溃
"""
import random, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch, torch.optim as optim
import torchvision.transforms as T
from tqdm import tqdm

from networks import DeepConvSurv, NegativeLogLikelihood, c_index_torch

warnings.filterwarnings("ignore", category=UserWarning)

# ----------------- 常量与路径 -----------------
SEED = 1
random.seed(SEED)
torch.manual_seed(SEED)

ROOT        = Path(__file__).resolve().parent
EXPAND_CSV  = ROOT / "cluster_result" / "patches_1000_cls10_expanded.csv"
MODEL_DIR   = ROOT / "log" / "wsisa_patch10" / "convimgmodel"
SEL_FILE    = ROOT / "log" / "selected_clusters.txt"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS     = 3          # 只跑几轮即可
BATCH_SIZE = 64
LR         = 1e-4
C_THRESH   = 0.0        # 只想有输出，阈值随便
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEAN, STD = [0.7, 0.6, 0.67], [0.26, 0.29, 0.25]
TRANSF = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])

# ----------------- 读取数据 -----------------
df = pd.read_csv(EXPAND_CSV)
print(f"[INFO] 总 patch 数: {len(df)}")
print("[INFO] 每簇 patch 数:")
for c, g in df.groupby("cluster"):
    print(f"  cluster {c}: {len(g)}")

clusters = sorted(df["cluster"].unique())
fold_cidx = {c: [] for c in clusters}   # 只有一折，仍按 list 存

# ------------- 划分训练 / 验证 / 测试 -------------
all_pids = df["pid"].unique().tolist()

# 挑 1 个 status=1 的病人做验证；如果没有，就随便选
event_pids = df[df["status"] == 1]["pid"].unique().tolist()
valid_pid  = random.choice(event_pids) if event_pids else random.choice(all_pids)
train_pids = [p for p in all_pids if p != valid_pid]
test_pids  = all_pids                   # 这里测试集就等于全体，目的是打印

print("\n========== 单折划分 ==========")
print("Train pids :", train_pids)
print("Valid pid  :", valid_pid)
print("Test  pids :", test_pids)

def idx_of(pids):
    return df.index[df["pid"].isin(pids)].tolist()

train_idx = idx_of(train_pids)
valid_idx = idx_of([valid_pid])
test_idx  = idx_of(test_pids)

def make_dataset(rows):
    xs, ts, es = [], [], []
    for _, r in rows.iterrows():
        xs.append(TRANSF(Image.open(ROOT / r["patch_path"])))
        ts.append(float(r["surv"]))
        es.append(float(r["status"]))
    return (
        torch.stack(xs),
        torch.tensor(ts, dtype=torch.float32),
        torch.tensor(es, dtype=torch.float32),
    )

# ----------------- 训练每个簇 -----------------
for c in tqdm(clusters, desc="train clusters"):
    tr = df.loc[train_idx].query("cluster == @c")
    va = df.loc[valid_idx].query("cluster == @c")
    te = df.loc[test_idx ].query("cluster == @c")

    if len(tr) < 5:                     # 太少就跳过
        print(f"[Skip] cluster {c} 训练样本 <5")
        fold_cidx[c].append(0.0)
        continue

    Xtr, Ttr, Etr = make_dataset(tr)
    Xva, Tva, Eva = make_dataset(va)
    Xte, Tte, Ete = make_dataset(te)

    model = DeepConvSurv(get_features=False).to(DEVICE)
    crit  = NegativeLogLikelihood()
    opt   = optim.Adam(model.parameters(), lr=LR)

    for ep in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(Xtr.size(0))
        for i in range(0, len(perm), BATCH_SIZE):
            idx = perm[i : i + BATCH_SIZE]
            loss = crit(model(Xtr[idx].to(DEVICE)),
                        Ttr[idx].to(DEVICE),
                        Etr[idx].to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()

    # --------- 计算验证 / 测试 C-index ---------
    model.eval()
    with torch.no_grad():
        try:
            vc = c_index_torch(model(Xva.to(DEVICE)), Tva, Eva)
        except ZeroDivisionError:
            vc = 0.0
            print(f"[Warn] cluster {c}: 验证集无可比较对，vc=0")
        try:
            tc = c_index_torch(model(Xte.to(DEVICE)), Tte, Ete)
        except ZeroDivisionError:
            tc = 0.0
            print(f"[Warn] cluster {c}: 测试集无可比较对，tc=0")

    fold_cidx[c].append(tc)
    print(f"cluster {c}: valid C={vc:.3f}, test C={tc:.3f}")

    torch.save({"model": model.state_dict()},
               MODEL_DIR / f"convimgmodel_cluster{c}_fold1.pth")

# ----------------- 选簇并保存 -----------------
selected = [c for c, scores in fold_cidx.items()
            if np.mean(scores) >= C_THRESH]
print("\n>>> 满足阈值的簇列表：", selected)
with open(SEL_FILE, "w") as f:
    f.write(",".join(map(str, selected)))
print(f"[Saved] selected clusters → {SEL_FILE}")