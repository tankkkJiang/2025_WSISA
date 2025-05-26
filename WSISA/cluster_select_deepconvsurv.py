#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WSISA/cluster_select_deepconvsurv.py

Step‑3  : 训练每个簇的 DeepConvSurv, 计算 C‑index, 选出“判别力强”的簇
使用 Leave‑One‑Patient‑Out(7 折) 而非原 5‑fold
权重保存至 log/wsisa_patch10/convimgmodel/convimgmodel_cluster{c}_fold{f}.pth
选簇列表写入 log/selected_clusters.txt (逗号分隔)
"""
import os, random, time, warnings, json
from pathlib import Path
from itertools import groupby

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T

from networks import DeepConvSurv, NegativeLogLikelihood, c_index_torch

warnings.filterwarnings("ignore", category=UserWarning)

# ============= 路径与常量 =============

SEED = 1
random.seed(SEED)
torch.manual_seed(SEED)

ROOT           = Path(__file__).resolve().parent
PATCH_CSV      = ROOT / "cluster_result" / "patches_1000_cls10.csv"
PATIENT_CSV    = ROOT / "data" / "patients.csv"
MODEL_DIR      = ROOT / "log" / "wsisa_patch10" / "convimgmodel"
SEL_FILE       = ROOT / "log" / "selected_clusters.txt"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS      = 8
BATCH_SIZE  = 64
LR          = 1e-4
C_THRESH    = 0.50           # C-index 阈值, 选簇用
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEAN = [0.6964, 0.5905, 0.6692]   # 可按数据集重新计算
STD  = [0.2559, 0.2943, 0.2462]
TRANSF = T.Compose([T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])


# ============= 数据准备 =============
patch_df    = pd.read_csv(PATCH_CSV)
patient_df  = pd.read_csv(PATIENT_CSV)
# 只取唯一 pid 列表，自动支持任意数量的 LOPO
unique_pids = patient_df["pid"].unique().tolist()
n_patients  = len(unique_pids)
print(f">>> 找到 {n_patients} 位唯一患者，用于 LOPO 折数")

# Helper: 取某几位病人的 patch 行 index
def idx_of(pids):
    return patch_df.index[patch_df["pid"].isin(pids)].tolist()


# ============= LOPO 训练 ============
clusters = sorted(patch_df["cluster"].unique())
fold_cidx  = {c: [] for c in clusters}      # 每簇各折 C-index

clusters  = sorted(patch_df["cluster"].unique())
fold_cidx = {c: [] for c in clusters}      # 每簇各折 C-index

# 用 unique_pids 做 LOPO
for fold, test_pid in enumerate(unique_pids, start=1):
    test_pids   = [test_pid]
    train_pids = [pid for pid in unique_pids if pid not in test_pids]
    # 再从 train_pids 随机挑 1 位作 valid，其余作 train
    random.shuffle(train_pids)
    valid_pids  = [train_pids.pop()]        # 剩 5 病人 train
    print(f"\n========== Fold {fold} / 7  ==========")
    print(f"  Train pids: {train_pids}")
    print(f"  Valid pids: {valid_pids}")
    print(f"  Test  pids: {test_pids}")

    train_idx = idx_of(train_pids)
    valid_idx = idx_of(valid_pids)
    test_idx  = idx_of(test_pids)

    # -------- 训练每个簇 --------
    for c in clusters:
        # 当前簇在三集合中的 patch index
        tr_rows = patch_df.loc[train_idx].query("cluster == @c")
        va_rows = patch_df.loc[valid_idx].query("cluster == @c")
        te_rows = patch_df.loc[test_idx].query("cluster == @c")

        if len(tr_rows) < 10:   # 训练样本过少直接跳过
            print(f"[Skip] cluster {c} has <10 train patches in fold‑{fold}")
            fold_cidx[c].append(0.0)
            continue

        # --- Dataset & Loader ---
        def make_dataset(rows):
            xs, ts, es = [], [], []
            for _, r in rows.iterrows():
                img = TRANSF(Image.open(ROOT / r["patch_path"]))
                xs.append(img)
                # 生存标签来自 patients.csv
                pinfo = patient_df.set_index("pid").loc[r["pid"]]
                ts.append(float(pinfo["surv"]))
                es.append(int(pinfo["status"]))
            xs = torch.stack(xs)
            ts = torch.tensor(ts, dtype=torch.float32)
            es = torch.tensor(es, dtype=torch.float32)
            return xs, ts, es

        Xtr, Ttr, Etr = make_dataset(tr_rows)
        Xva, Tva, Eva = make_dataset(va_rows)
        Xte, Tte, Ete = make_dataset(te_rows)

        model = DeepConvSurv(get_features=False).to(DEVICE)
        crit  = NegativeLogLikelihood()
        opt   = optim.Adam(model.parameters(), lr=LR)

        best_c = 0.0
        for ep in range(1, EPOCHS + 1):
            model.train()
            perm = torch.randperm(Xtr.size(0))
            for s in range(0, len(perm), BATCH_SIZE):
                idx = perm[s:s+BATCH_SIZE]
                x   = Xtr[idx].to(DEVICE)
                t   = Ttr[idx].to(DEVICE)
                e   = Etr[idx].to(DEVICE)

                risk = model(x)
                loss = crit(risk, t, e)
                opt.zero_grad()
                loss.backward()
                opt.step()

            # ---- valid C‑index ----
            model.eval()
            with torch.no_grad():
                v_risk = model(Xva.to(DEVICE))
                v_c    = c_index_torch(v_risk, Tva, Eva)
            if v_c > best_c:
                best_c = v_c
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": opt.state_dict(),
                        "epoch": ep
                    },
                    MODEL_DIR / f"convimgmodel_cluster{c}_fold{fold}.pth"
                )
        # ---- test C‑index ----
        model.eval()
        with torch.no_grad():
            te_risk = model(Xte.to(DEVICE))
            te_c    = c_index_torch(te_risk, Tte, Ete)
        fold_cidx[c].append(te_c)
        print(f"  [Fold‑{fold}] cluster {c:2d}: best_valid C={best_c:.3f}, test C={te_c:.3f}")

# ============= 选簇 =============
selected = []
print("\n>>> 按簇汇总 (7 折均值) C‑index :")
for c in clusters:
    scores = fold_cidx[c]
    mean_c = np.mean(scores)
    print(f"Cluster {c:2d}: mean C‑index = {mean_c:.4f}")
    if mean_c >= C_THRESH:
        selected.append(c)

print("\n>>> 选中簇:", selected)
with open(SEL_FILE, "w") as f:
    f.write(",".join(map(str, selected)))
print(f"[Saved] selected cluster list → {SEL_FILE}")