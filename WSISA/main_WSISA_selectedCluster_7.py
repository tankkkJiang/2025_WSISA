#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WSISA/main_WSISA_selectedCluster_7.py

Step‑4 (Aggregation):
  - 读取已选簇的 patch‑level 模型
  - 将 patch → patient (加权平均)
  - 导出 train / valid / test 三集合的患者级特征 & 风险
"""
import random
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import torchvision.transforms as T

from networks import DeepConvSurv
from utils.WSISA_utils import patient_features   # 已在 utils 里修过 import

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------- paths & config ----------------
ROOT        = Path(__file__).resolve().parent
PATCH_CSV   = ROOT / "cluster_result" / "patches_1000_cls10_expanded.csv"   # ← 用扩展后的 CSV
MODEL_DIR   = ROOT / "log"    / "wsisa_patch10" / "convimgmodel"
SEL_FILE    = ROOT / "log"    / "selected_clusters.txt"
OUT_DIR     = ROOT / "results"
OUT_DIR.mkdir(exist_ok=True, parents=True)

print("正在打印配置参数：")
print(f"  PATCH_CSV    路径: {PATCH_CSV}")
print(f"  MODEL_DIR    路径: {MODEL_DIR}")
print(f"  SEL_FILE     路径: {SEL_FILE}")
print(f"  OUT_DIR      路径: {OUT_DIR}")

# ---------------- 读取 CSV ----------------
assert SEL_FILE.exists(), "错误：请先运行 cluster_select_deepconvsurv.py 完成选簇"
selected_cluster = [int(x) for x in SEL_FILE.read_text().split(",") if x.strip().isdigit()]
print(f">>> 已选簇 (共 {len(selected_cluster)} 个): {selected_cluster}")

try:
    patch_df = pd.read_csv(PATCH_CSV)
    print(f">>> 载入 patch CSV，行数: {len(patch_df)}")
except Exception as e:
    raise RuntimeError(f"错误：无法读取 {PATCH_CSV}: {e}")

# 必要字段检测
for col in ["pid", "surv", "status", "patch_path", "cluster"]:
    if col not in patch_df.columns:
        raise KeyError(f"CSV 缺少必须列: '{col}'")

# 从 patch_df 里抽取患者级信息
patient_df = patch_df[["pid", "surv", "status"]].drop_duplicates()
print(f">>> 患者总数: {len(patient_df)}")

# ---------------- 图像预处理 ----------------
MEAN = [0.6964, 0.5905, 0.6692]
STD  = [0.2559, 0.2943, 0.2462]
print(f">>> 图像归一化参数：MEAN={MEAN}, STD={STD}")
TRANSF = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD)
])

# ------------- helper -------------
def make_patch_dataset(rows):
    """返回 (Tensor[N,C,H,W], list_of_pid)；若 rows 为空返回 (None,[])"""
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


def agg_weighted_features(patch_df: pd.DataFrame,
                          sel_clusters,
                          fea_dim: int) -> pd.DataFrame:
    """
    显式实现论文里的加权平均 x_{ij} = w_{ij} * mean_k x_{ijk}
    返回列: pid, surv, status, fea_0 ... fea_{J*fea_dim-1}
    """
    patients = patch_df["pid"].unique().tolist()
    rows_out = []

    for pid in patients:
        df_p = patch_df[patch_df["pid"] == pid]
        total_patches = len(df_p)
        row = {
            "pid":    pid,
            "surv":   df_p["surv"].iloc[0],
            "status": df_p["status"].iloc[0]
        }
        # 记录每个簇的权重 w_ij 用于可视化
        weight_log = {}

        feat_all = []
        for c in sel_clusters:
            df_pc = df_p[df_p["cluster"] == c]
            n_c   = len(df_pc)
            if n_c == 0:
                weight = 0.0
                feat_c = torch.zeros(fea_dim)
            else:
                weight = n_c / total_patches
                feat_mat = torch.tensor(
                    df_pc.loc[:, [f"fea_{i}" for i in range(fea_dim)]].values,
                    dtype=torch.float32
                )
                feat_c = feat_mat.mean(dim=0)
            weight_log[c] = round(weight, 4)
            feat_all.extend((weight * feat_c).tolist())

        # 打印前 3 个患者的权重分布
        if len(rows_out) < 3:
            print(f"    [Debug] patient {pid} 的 w_ij: {weight_log}")

        for i, val in enumerate(feat_all):
            row[f"fea_{i}"] = val
        rows_out.append(row)

    return pd.DataFrame(rows_out)

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
            for c in tqdm(selected_cluster,
                          desc=f"[Fold{fold}-{split_name}] 簇推理",
                          leave=False,
                          ncols=80):
                mpath = MODEL_DIR / f"convimgmodel_cluster{c}_fold{fold}.pth"
                if not mpath.exists():          # 若该折无模型→回退 fold1
                    mpath = MODEL_DIR / f"convimgmodel_cluster{c}_fold1.pth"
                if not mpath.exists():
                    print(f"[Warn] 模型文件不存在: {mpath}")
                    continue

                # 加载模型
                try:
                    model = DeepConvSurv(get_features=True)
                    model.load_state_dict(torch.load(mpath, map_location="cpu")["model"])
                    model.eval()
                except Exception as e:
                    print(f"[Error] 加载模型 {mpath.name} 失败: {e}")
                    continue

                # 取属于该簇的 patch
                idxs = patch_feat_df.index[patch_feat_df["cluster"] == c].tolist()
                if not idxs:
                    print(f"    cluster {c}: 无 patch，跳过")
                    continue

                X, _ = make_patch_dataset(patch_feat_df.loc[idxs])
                if X is None:
                    print(f"    cluster {c}: X 张量为空，跳过")
                    continue

                # 推理
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

        # patch → patient 加权
        try:
            # 用新实现替换老 patient_features；若想对比二者可切换
            fea_dim_here = feat_np.shape[1]        # 由上一循环得到
            patient_fea  = agg_weighted_features(patch_feat_df,
                                                 selected_cluster,
                                                 fea_dim_here)
            patient_risk = agg_weighted_features(patch_risk_df,
                                                 selected_cluster,
                                                 fea_dim=1)
            print(f"    [Info] 聚合完成: 每位患者特征维度 = "
                  f"{len(selected_cluster)*fea_dim_here}")
        except Exception as e:
            print(f"[Error] 聚合 patient_features 失败: {e}")
            return

        out_fea = OUT_DIR / f"{split_name}_patient_features_fold{fold}.csv"
        out_rsk = OUT_DIR / f"{split_name}_patient_risks_fold{fold}.csv"
        try:
            patient_fea.to_csv(out_fea, index=False)
            patient_risk.to_csv(out_rsk, index=False)
            print(f"[Success] {split_name} 保存特征: {out_fea.name}, 风险: {out_rsk.name}")
        except Exception as e:
            print(f"[Error] 保存 CSV 失败: {e}")

# ---------------- LOPO‑7 Aggregation ----------------
all_pids = patient_df["pid"].tolist()
print(f"\n>>> LOPO-7 开始，共 {len(all_pids)} 折")
for fold, test_pid in tqdm(list(enumerate(all_pids, start=1)),
                           desc="LOPO-7 折数",
                           ncols=80):
    test_pids  = [test_pid]
    train_pids = [pid for pid in all_pids if pid != test_pid]
    random.shuffle(train_pids)
    valid_pids = [train_pids.pop()]   # 单个患者做 valid
    aggregate_one_fold(fold, train_pids, valid_pids, test_pids)