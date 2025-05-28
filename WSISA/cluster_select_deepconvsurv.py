#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WSISA/cluster_select_deepconvsurv.py

目的：在多位病人的数据集上完整运行，
      且保证测试集中既有生存（status=1）也有死亡（status=0）的患者：
      * 每个簇训练 1 个 DeepConvSurv
      * 测试集 5 患者
      * 训练/验证集 为剩余患者
      * 针对 C-index=0 或 C-index=0.5 的簇打印原因
      * 只保留 C-index >= 0.5 的簇
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

# 只保留 C-index >= 0.5 的簇
C_THRESH = 0.5

MEAN = [0.6964, 0.5905, 0.6692]
STD  = [0.2559, 0.2943, 0.2462]
TRANSF = T.Compose([T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])

# ---------------- 读取并打印基本信息 ----------------
df = pd.read_csv(PATCH_CSV)

print("\n>>> cluster 分布:")
for cid, g in df.groupby("cluster"):
    print(f"  cluster {cid:2d}: {len(g):6d} patches")

unique_pids = df["pid"].unique().tolist()
print(f"\n>>> 数据集共有 {len(unique_pids)} 位病人\n")

# 每位患者的 status 分布
pid_status = df.groupby("pid")["status"].apply(lambda x: sorted(x.unique())).to_dict()
print(">>> 各患者生存状态:")
for pid, sts in pid_status.items():
    print(f"  {pid}: {sts}")

# 自动挑选测试集 5 患者，保证包含生存和死亡
test_pids = None
for combo in combinations(unique_pids, 5):
    statuses = set()
    for pid in combo:
        statuses.update(pid_status[pid])
    if statuses == {0, 1}:
        test_pids = list(combo)
        break
if test_pids is None:
    raise RuntimeError("找不到包含两种生存状态的 5 患者组合")

train_val_pids = [pid for pid in unique_pids if pid not in test_pids]
print(f"\n>>> 测试集患者（5 位）：{test_pids}")
print(f">>> 训练/验证集患者（{len(train_val_pids)} 位）：{train_val_pids}\n")

# ---------------- 构造数据集 ----------------
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

print(f"========== Fold 1 / 1 (train={len(train_val_pids)}, test=5) ==========")
for c in tqdm(clusters, desc="Clusters"):
    df_c  = df.query("cluster == @c")
    df_tv = df_c[df_c.pid.isin(train_val_pids)]
    df_te = df_c[df_c.pid.isin(test_pids)]

    # 样本太少
    if len(df_tv) < 5:
        print(f"[Skip] cluster {c}: 训练集样本 <5")
        fold_cidx[c] = 0.0
        continue

    # 载入 train+val
    X_tv, T_tv, E_tv = make_dataset(df_tv)
    if X_tv is None:
        print(f"[Skip] cluster {c}: 未加载到训练数据")
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
            xb, tb, eb = X_tv[idx].to(DEVICE), T_tv[idx].to(DEVICE), E_tv[idx].to(DEVICE)
            loss = crit(model(xb), tb, eb)
            opt.zero_grad(); loss.backward(); opt.step()

    # 载入 test
    X_te, T_te, E_te = make_dataset(df_te)
    if X_te is None:
        print(f"[Warn] cluster {c}: 测试数据为空，C-index 置 0")
        c_idx = 0.0
    else:
        model.eval()
        scores_list = []
        with torch.no_grad():
            for i in range(0, X_te.size(0), BATCH_SIZE):
                xb = X_te[i : i + BATCH_SIZE].to(DEVICE)
                scores_list.append(model(xb).cpu())
        all_scores = torch.cat(scores_list, dim=0)
        try:
            c_idx = c_index_torch(all_scores, T_te, E_te)
        except ZeroDivisionError:
            c_idx = 0.0

    fold_cidx[c] = c_idx
    print(f"Cluster {c:2d}  →  C-index = {c_idx:.4f}")
    if c_idx == 0.0:
        print(f"  [Info] cluster {c}: C-index=0.0 可能因为测试集中无可比较生存对或所有status一致")

    # 保存模型
    torch.save(
        {"model": model.state_dict()},
        MODEL_DIR / f"convimgmodel_cluster{c}_fold1.pth"
    )

# ---------------- 选簇并保存列表 ----------------
selected = [c for c, v in fold_cidx.items() if v >= C_THRESH]
print("\n>>> C_THRESH =", C_THRESH)
print(">>> 满足阈值的簇：", selected)

with open(SEL_FILE, "w") as f:
    f.write(",".join(map(str, selected)))
print(f"[Saved] selected clusters → {SEL_FILE}")