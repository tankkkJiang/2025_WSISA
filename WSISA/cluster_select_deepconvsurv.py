#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WSISA/cluster_select_deepconvsurv.py

Step-3  : 训练每个簇的 DeepConvSurv, 计算 C-index, 选出“判别力强”的簇
使用 Leave-One-Patient-Out (LOPO) 而非原 5-fold
权重保存至 log/wsisa_patch10/convimgmodel/convimgmodel_cluster{c}_fold{f}.pth
选簇列表写入 log/selected_clusters.txt (逗号分隔)
"""
import os, random, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch, torch.optim as optim
import torchvision.transforms as T
from tqdm import tqdm

from networks import DeepConvSurv, NegativeLogLikelihood, c_index_torch

warnings.filterwarnings("ignore", category=UserWarning)

# ============= 路径与常量 =============
SEED       = 1
random.seed(SEED)
torch.manual_seed(SEED)

ROOT       = Path(__file__).resolve().parent
# 直接读取已扩展的 patch CSV，包含 patch_path, pid, surv, status, cluster
EXPAND_CSV = ROOT / "cluster_result" / "patches_1000_cls10_expanded.csv"
MODEL_DIR  = ROOT / "log" / "wsisa_patch10" / "convimgmodel"
SEL_FILE   = ROOT / "log" / "selected_clusters.txt"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS     = 8
BATCH_SIZE = 64
LR         = 1e-4
C_THRESH   = 0.50           # C-index 阈值, 选簇用
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEAN = [0.6964, 0.5905, 0.6692]
STD  = [0.2559, 0.2943, 0.2462]
TRANSF = T.Compose([T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])

# ============= 数据准备 =============
# 读取扩展后的 patch + 标签表
df = pd.read_csv(EXPAND_CSV)
# 按 cluster 统计 patch 数
print("\n>>> Patch 数量统计 (按 cluster)：")
for c, g in df.groupby("cluster"):
    print(f"Cluster {c:2d}: {len(g):6d} patches")

# LOPO: N = 唯一患者数
unique_pids = df["pid"].unique().tolist()
print(f">>> 找到 {len(unique_pids)} 位唯一患者，用于 LOPO 折数")

# Helper: 取某几位病人的 patch 行 index
def idx_of(pids):
    return df.index[df["pid"].isin(pids)].tolist()

# ============= LOPO 训练 ============
clusters = sorted(df["cluster"].unique())
fold_cidx = {c: [] for c in clusters}

for fold, test_pid in enumerate(unique_pids, start=1):
    test_pids  = [test_pid]
    train_pids = [p for p in unique_pids if p not in test_pids]
    random.shuffle(train_pids)
    valid_pids = [train_pids.pop()]

    print(f"\n========== Fold {fold} / {len(unique_pids)} ==========")
    print(f"  Train: {train_pids}")
    print(f"  Valid: {valid_pids}")
    print(f"  Test : {test_pids}")

    train_idx = idx_of(train_pids)
    valid_idx = idx_of(valid_pids)
    test_idx  = idx_of(test_pids)

    def make_dataset(rows):
        """从 DataFrame rows 构造 (X, t, e)"""
        xs, ts, es = [], [], []
        for _, r in rows.iterrows():
            img = Image.open(ROOT / r["patch_path"])
            xs.append(TRANSF(img))
            ts.append(float(r["surv"]))
            es.append(float(r["status"]))
        return (
            torch.stack(xs),
            torch.tensor(ts, dtype=torch.float32),
            torch.tensor(es, dtype=torch.float32),
        )

    # 针对每个簇训练 & 测评
    for c in clusters:
        tr = df.loc[train_idx].query("cluster == @c")
        va = df.loc[valid_idx].query("cluster == @c")
        te = df.loc[test_idx].query("cluster == @c")

        if len(tr) < 10:
            print(f"[Skip] cluster {c} 在 fold-{fold} 训练集样本 <10，跳过")
            fold_cidx[c].append(0.0)
            continue

        Xtr, Ttr, Etr = make_dataset(tr)
        Xva, Tva, Eva = make_dataset(va)
        Xte, Tte, Ete = make_dataset(te)

        model = DeepConvSurv(get_features=False).to(DEVICE)
        crit  = NegativeLogLikelihood()
        opt   = optim.Adam(model.parameters(), lr=LR)

        best_vc = 0.0
        for ep in range(1, EPOCHS+1):
            model.train()
            perm = torch.randperm(Xtr.size(0))
            for i in range(0, len(perm), BATCH_SIZE):
                idx = perm[i : i+BATCH_SIZE]
                x   = Xtr[idx].to(DEVICE)
                t   = Ttr[idx].to(DEVICE)
                e   = Etr[idx].to(DEVICE)
                loss = crit(model(x), t, e)
                opt.zero_grad(); loss.backward(); opt.step()
            # 验证 C-index
            model.eval()
            with torch.no_grad():
                vc = c_index_torch(model(Xva.to(DEVICE)), Tva, Eva)
            if vc > best_vc:
                best_vc = vc
                torch.save(
                    {"model": model.state_dict()},
                    MODEL_DIR / f"convimgmodel_cluster{c}_fold{fold}.pth"
                )
        # 测试 C-index
        model.eval()
        with torch.no_grad():
            tc = c_index_torch(model(Xte.to(DEVICE)), Tte, Ete)
        fold_cidx[c].append(tc)
        print(f"  [Fold {fold}] cluster {c:2d}: valid C-index={best_vc:.3f}, test C-index={tc:.3f}")

# ============= 选簇 =============
selected = [c for c, lst in fold_cidx.items() if np.mean(lst) >= C_THRESH]
print("\n>>> 平均 test C-index ≥ %.2f 的簇：" % C_THRESH, selected)
with open(SEL_FILE, "w") as f:
    f.write(",".join(map(str, selected)))
print(f"[Saved] selected cluster list → {SEL_FILE}")