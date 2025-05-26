#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WSISA/cluster_select_deepconvsurv.py

目的：在只有 7 位病人的数据集上也能完整跑一遍，
      且保证测试集中既有生存（status=1）也有死亡（status=0）的患者：
      * 每个簇各自训练 1 个 DeepConvSurv（训练/验证集共 5 患者）
      * 测试集 2 患者
      * 打印训练过程和 C-index（若无可比较样本对则记 0.0）
"""
import random
import warnings
from itertools import combinations
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

# ---------------- 配置 ----------------
SEED       = 1
random.seed(SEED)
torch.manual_seed(SEED)

ROOT       = Path(__file__).resolve().parent
PATCH_CSV  = ROOT / "cluster_result" / "patches_1000_cls10_expanded.csv"
MODEL_DIR  = ROOT / "log" / "wsisa_patch10" / "convimgmodel"
SEL_FILE   = ROOT / "log" / "selected_clusters.txt"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS, BATCH_SIZE, LR = 3, 16, 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
C_THRESH = 0.0

MEAN=[0.6964,0.5905,0.6692]
STD =[0.2559,0.2943,0.2462]
TRANSF = T.Compose([T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])

# ---------------- 读取并打印数据集基本信息 ----------------
df = pd.read_csv(PATCH_CSV)

print("\n>>> cluster 分布:")
for cid, g in df.groupby("cluster"):
    print(f"  cluster {cid:2d}: {len(g):6d} patches")

unique_pids = df["pid"].unique().tolist()
print(f"\n>>> 数据集共有 {len(unique_pids)} 位病人\n")

# 打印每位患者的生存状态分布
pid_status = df.groupby("pid")["status"].apply(lambda x: sorted(x.unique())).to_dict()
print(">>> 各患者生存状态:")
for pid, sts in pid_status.items():
    print(f"  {pid}: {sts}")

# 挑选 test_pids：找一对能覆盖两种 status 的患者
test_pids = None
for combo in combinations(unique_pids, 2):
    statuses = df[df.pid.isin(combo)]["status"].unique()
    if set(statuses) == {0, 1}:
        test_pids = list(combo)
        break
if test_pids is None:
    raise RuntimeError("找不到包含两种生存状态的 2 患者组合")
train_val_pids = [pid for pid in unique_pids if pid not in test_pids]
print(f"\n>>> 测试集患者（2）：{test_pids}")
print(f">>> 训练/验证集患者（{len(train_val_pids)}）：{train_val_pids}\n")

# ---------------- 构造数据集函数 ----------------
def make_dataset(rows):
    xs, ts, es = [], [], []
    for _, r in tqdm(rows.iterrows(), total=len(rows), desc="  load patches", leave=False):
        try:
            img = Image.open(ROOT / r["patch_path"])
        except Exception as e:
            warnings.warn(f"[Warn] 无法打开 {r['patch_path']}: {e}", UserWarning)
            continue
        xs.append(TRANSF(img))
        ts.append(float(r["surv"]))
        es.append(float(r["status"]))
    if not xs:
        return None, None, None
    return torch.stack(xs), torch.tensor(ts), torch.tensor(es)

# ---------------- per-cluster 训练与测试 ----------------
clusters = sorted(df["cluster"].unique())
fold_cidx = {}

print("========== Fold 1 / 1 (train=5, test=2 patients) ==========")
for c in tqdm(clusters, desc="Clusters"):
    # 按 cluster 和 pid 划分 train_val / test
    df_c = df.query("cluster == @c")
    df_tv = df_c[df_c.pid.isin(train_val_pids)]
    df_te = df_c[df_c.pid.isin(test_pids)]

    if len(df_tv) < 5:
        print(f"[Skip] cluster {c} 训练集样本 <5，跳过")
        fold_cidx[c] = 0.0
        continue

    # 载入 train_val 数据
    X_tv, T_tv, E_tv = make_dataset(df_tv)
    if X_tv is None:
        print(f"[Skip] cluster {c} 未加载到 train 数据，跳过")
        fold_cidx[c] = 0.0
        continue

    # 训练
    model = DeepConvSurv(get_features=False).to(DEVICE)
    crit  = NegativeLogLikelihood()
    opt   = optim.Adam(model.parameters(), lr=LR)

    for ep in range(1, EPOCHS+1):
        model.train()
        perm = torch.randperm(X_tv.size(0))
        for i in range(0, len(perm), BATCH_SIZE):
            idx = perm[i : i + BATCH_SIZE]
            xb = X_tv[idx].to(DEVICE)
            tb = T_tv[idx].to(DEVICE)
            eb = E_tv[idx].to(DEVICE)
            loss = crit(model(xb), tb, eb)
            opt.zero_grad()
            loss.backward()
            opt.step()

    # 载入 test 数据
    X_te, T_te, E_te = make_dataset(df_te)
    if X_te is None:
        print(f"[Warn] cluster {c} 无法加载测试数据，C-index 置 0")
        fold_cidx[c] = 0.0
    else:
        # 测试分批推理，防 OOM
        model.eval()
        all_scores = []
        with torch.no_grad():
            for i in range(0, X_te.size(0), BATCH_SIZE):
                xb = X_te[i : i + BATCH_SIZE].to(DEVICE)
                scores = model(xb).cpu()
                all_scores.append(scores)
        all_scores = torch.cat(all_scores, dim=0)
        try:
            c_idx = c_index_torch(all_scores, T_te, E_te)
        except ZeroDivisionError:
            c_idx = 0.0
        fold_cidx[c] = c_idx
        print(f"Cluster {c:2d}  完成  →  C-index = {c_idx:.4f}")

    # 保存模型
    torch.save(
        {"model": model.state_dict()},
        MODEL_DIR / f"convimgmodel_cluster{c}_fold1.pth"
    )

# ---------------- 选簇并保存列表 ----------------
selected = [c for c, v in fold_cidx.items() if v >= C_THRESH]
print("\n>>> 满足阈值的簇：", selected)
with open(SEL_FILE, "w") as f:
    f.write(",".join(map(str, selected)))
print(f"[Saved] selected cluster list → {SEL_FILE}")