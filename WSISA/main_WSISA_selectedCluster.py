#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WSISA/main_WSISA_selectedCluster.py

Step‑4 (Aggregation):
  ‑ 读取已选簇的 patch‑level 模型
  ‑ 将 patch → patient (加权平均)
  ‑ 导出 train / valid / test 三集合的患者级特征 & 风险
"""
import os, random, json
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T

from utils.WSISA_utils import patient_features
from WSISA_dataloader import WSISA_get_feat_dataloader
from networks import DeepConvSurv   # 纯 PyTorch 版
# --------------------------------------------------

ROOT        = Path(__file__).resolve().parent
PATCH_CSV   = ROOT / "cluster_result" / "patches_1000_cls10.csv"
PATIENT_CSV = ROOT / "data" / "patients.csv"
MODEL_DIR   = ROOT / "log" / "wsisa_patch10" / "convimgmodel"
SEL_FILE    = ROOT / "log" / "selected_clusters.txt"
OUT_DIR     = ROOT / "results"
OUT_DIR.mkdir(exist_ok=True, parents=True)

assert SEL_FILE.exists(), "请先运行 cluster_select_deepconvsurv_pytorch.py 完成选簇"
selected_cluster = [int(x) for x in SEL_FILE.read_text().strip().split(",")]
print(">>> 将在聚合阶段使用的簇:", selected_cluster)

patch_df   = pd.read_csv(PATCH_CSV)
patient_df = pd.read_csv(PATIENT_CSV)
MEAN = [0.6964, 0.5905, 0.6692]
STD  = [0.2559, 0.2943, 0.2462]
TRANSF = T.Compose([T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])


# ------------- helper -------------
def make_patch_dataset(rows):
    xs, pids = [], []
    for _, r in rows.iterrows():
        xs.append(TRANSF(Image.open(ROOT / r["patch_path"])))
        pids.append(r["pid"])
    xs = torch.stack(xs)
    return xs, pids


def aggregate_one_fold(fold, train_pids, valid_pids, test_pids):
    print(f"\n========== [Aggregation] Fold {fold} ==========")
    splits = {"train": train_pids, "valid": valid_pids, "test": test_pids}
    for sp, pids in splits.items():
        rows = patch_df[patch_df["pid"].isin(pids)]
        out_csv_fea  = OUT_DIR / f"{sp}_patient_features_fold{fold}.csv"
        out_csv_risk = OUT_DIR / f"{sp}_patient_risks_fold{fold}.csv"

        # 1. 取每个 selected cluster 的模型
        model_paths = {
            c: MODEL_DIR / f"convimgmodel_cluster{c}_fold{fold}.pth"
            for c in selected_cluster
        }
        # 2. 模型推理: patch → 特征 / 风险
        patch_feat_df = rows.copy()
        patch_risk_df = rows.copy()
        patch_feat_df[["fea_0"]]  = 0.0  # 占位；列稍后整体替换
        patch_risk_df[["fea_0"]]  = 0.0

        with torch.no_grad():
            for c in selected_cluster:
                mpath = model_paths[c]
                model = DeepConvSurv(get_features=True)
                model.load_state_dict(torch.load(mpath, map_location="cpu")["model"])
                model.eval()

                rows_c = patch_feat_df.query("cluster == @c")
                if rows_c.empty:
                    continue
                X, idx_pids = make_patch_dataset(rows_c)
                feat, risk  = model(X)
                patch_feat_df.loc[rows_c.index, "fea_0":"fea_31"] = feat.numpy()
                patch_risk_df.loc[rows_c.index, "fea_0"]          = risk.numpy()

        # 3. patch → patient 加权
        patient_fea  = patient_features(patch_feat_df, selected_cluster)
        patient_risk = patient_features(patch_risk_df, selected_cluster, fea_dim=1)
        patient_fea.to_csv(out_csv_fea,  index=False)
        patient_risk.to_csv(out_csv_risk, index=False)
        print(f"[{sp}] {len(pids)} patients  →  特征:{out_csv_fea.name}  风险:{out_csv_risk.name}")


# ============= LOPO‑7 Aggregation =============
all_pids = patient_df["pid"].tolist()
for fold, test_pid in enumerate(all_pids, start=1):
    test_pids   = [test_pid]
    train_pids  = all_pids.copy(); train_pids.remove(test_pid)
    random.shuffle(train_pids)
    valid_pids  = [train_pids.pop()]     # 1 病人 valid
    aggregate_one_fold(fold, train_pids, valid_pids, test_pids)