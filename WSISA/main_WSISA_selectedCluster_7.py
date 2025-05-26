#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WSISA/main_WSISA_selectedCluster_7.py

Step-4 (Aggregation):
  - 读取已选簇的 patch-level 模型
  - 将 patch → patient (加权平均)
  - 导出 train / valid / test 三集合的患者级特征 & 风险
"""
import os
import random
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torchvision.transforms as T

from networks import DeepConvSurv
from utils.WSISA_utils import patient_features

# ---------------- paths & config ----------------
ROOT        = Path(__file__).resolve().parent
PATCH_CSV   = ROOT / "cluster_result" / "patches_1000_cls10.csv"
PATIENT_CSV = ROOT / "data"   / "patients.csv"
MODEL_DIR   = ROOT / "log"    / "wsisa_patch10" / "convimgmodel"
SEL_FILE    = ROOT / "log"    / "selected_clusters.txt"
OUT_DIR     = ROOT / "results"
OUT_DIR.mkdir(exist_ok=True, parents=True)

print("正在打印配置参数：")
print(f"  PATCH_CSV  路径: {PATCH_CSV}")
print(f"  PATIENT_CSV 路径: {PATIENT_CSV}")
print(f"  MODEL_DIR   路径: {MODEL_DIR}")
print(f"  SEL_FILE    路径: {SEL_FILE}")
print(f"  OUT_DIR     路径: {OUT_DIR}")

assert SEL_FILE.exists(), "错误：请先运行 cluster_select_deepconvsurv.py 完成选簇"
selected_cluster = [int(x) for x in SEL_FILE.read_text().split(",") if x.strip().isdigit()]
print(f">>> 已选簇 (共 {len(selected_cluster)} 个): {selected_cluster}")

try:
    patch_df   = pd.read_csv(PATCH_CSV)
    print(f">>> 载入 patch CSV，行数: {len(patch_df)}")
except Exception as e:
    raise RuntimeError(f"错误：无法读取 {PATCH_CSV}: {e}")

try:
    patient_df = pd.read_csv(PATIENT_CSV)
    print(f">>> 载入 patient CSV，行数: {len(patient_df)}")
except Exception as e:
    raise RuntimeError(f"错误：无法读取 {PATIENT_CSV}: {e}")

MEAN = [0.6964, 0.5905, 0.6692]
STD  = [0.2559, 0.2943, 0.2462]
print(f">>> 图像归一化参数：MEAN={MEAN}, STD={STD}")
TRANSF = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD)
])

# ------------- helper -------------
def make_patch_dataset(rows):
    """返回 (Tensor[N,C,H,W], list_of_pids)"""
    xs, pids = [], []
    for _, r in rows.iterrows():
        try:
            img = Image.open(ROOT / r["patch_path"])
        except Exception as e:
            print(f"[Warn] 无法打开图像 {r['patch_path']}: {e}")
            continue
        xs.append(TRANSF(img))
        pids.append(r["pid"])
    if not xs:
        return None, []
    return torch.stack(xs), pids

# ---------------- aggregation per fold ----------------
def aggregate_one_fold(fold, train_pids, valid_pids, test_pids):
    print(f"\n========== 开始第 {fold} 折聚合 ==========")
    splits = {"train": train_pids, "valid": valid_pids, "test": test_pids}

    for split_name, pids in splits.items():
        print(f"\n--- 处理 {split_name} 集合: 共 {len(pids)} 位患者 ---")
        rows = patch_df[patch_df["pid"].isin(pids)].copy()
        print(f"    共载入 {len(rows)} 条 patch 记录")

        # 准备 DataFrames
        patch_feat_df = rows.reset_index(drop=True).copy()
        patch_risk_df = rows.reset_index(drop=True).copy()
        patch_risk_df["risk"] = 0.0

        with torch.no_grad():
            for c in selected_cluster:
                mpath = MODEL_DIR / f"convimgmodel_cluster{c}_fold{fold}.pth"
                if not mpath.exists():
                    print(f"[Warn] 模型文件不存在: {mpath}")
                    continue
                try:
                    model = DeepConvSurv(get_features=True)
                    ckpt  = torch.load(mpath, map_location="cpu")
                    model.load_state_dict(ckpt["model"])
                    model.eval()
                except Exception as e:
                    print(f"[Error] 加载模型 cluster {c}, fold {fold} 失败: {e}")
                    continue

                idxs = patch_feat_df.index[patch_feat_df["cluster"] == c].tolist()
                if not idxs:
                    print(f"    cluster {c}: 无 patch，跳过")
                    continue

                sub_df = patch_feat_df.loc[idxs]
                X, _  = make_patch_dataset(sub_df)
                if X is None:
                    print(f"    cluster {c}: X 张量为空，跳过")
                    continue

                try:
                    feat, risk = model(X)
                except Exception as e:
                    print(f"[Error] 模型推理失败 cluster {c}: {e}")
                    continue

                feat_np = feat.cpu().numpy()
                risk_np = risk.cpu().numpy()
                D = feat_np.shape[1]
                feat_cols = [f"fea_{i}" for i in range(D)]
                for col in feat_cols:
                    if col not in patch_feat_df.columns:
                        patch_feat_df[col] = 0.0

                patch_feat_df.loc[idxs, feat_cols] = feat_np
                patch_risk_df.loc[idxs, "risk"] = risk_np
                print(f"    cluster {c}: 推理完成, 特征维度={D}, patch 数={len(idxs)}")

        # patch -> patient 加权
        try:
            patient_fea  = patient_features(patch_feat_df, selected_cluster)
            patient_risk = patient_features(patch_risk_df, selected_cluster, fea_dim=1)
        except Exception as e:
            print(f"[Error] 聚合 patient_features 失败: {e}")
            continue

        out_fea = OUT_DIR / f"{split_name}_patient_features_fold{fold}.csv"
        out_rsk = OUT_DIR / f"{split_name}_patient_risks_fold{fold}.csv"
        try:
            patient_fea.to_csv(out_fea, index=False)
            patient_risk.to_csv(out_rsk, index=False)
            print(f"[Success] {split_name} 保存特征: {out_fea.name}, 风险: {out_rsk.name}")
        except Exception as e:
            print(f"[Error] 保存 CSV 失败: {e}")

# ---------------- LOPO-7 Aggregation ----------------
all_pids = patient_df["pid"].tolist()
for fold, test_pid in enumerate(all_pids, start=1):
    test_pids  = [test_pid]
    train_pids = [pid for pid in all_pids if pid != test_pid]
    random.shuffle(train_pids)
    valid_pids = [train_pids.pop()]
    aggregate_one_fold(fold, train_pids, valid_pids, test_pids)