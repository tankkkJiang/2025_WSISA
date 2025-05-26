#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WSISA/main_WSISA_selectedCluster_1fold.py

Step-4 (Aggregation) 单折版:
  - 读取已选簇的 patch-level 模型
  - 将 patch → patient (加权平均)
  - 导出 train / valid / test 三集合的患者级特征 & 风险
  - 只运行一折，避免 LOPO-7 带来的长时间开销
"""
import random
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import torchvision.transforms as T

from networks import DeepConvSurv
from utils.WSISA_utils import patient_features   # 仍可用旧实现

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------- paths & config ----------------
ROOT        = Path(__file__).resolve().parent
PATCH_CSV   = ROOT / "cluster_result" / "patches_1000_cls10_expanded.csv"
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

def agg_weighted_features(patch_df, sel_clusters, fea_dim):
    """
    加权平均 x_{ij} = w_{ij} * mean_k x_{ijk}
    返回患者级 DataFrame（pid, surv, status, fea_...）
    """
    patients = patch_df["pid"].unique().tolist()
    out_rows = []
    for pid in patients:
        df_p = patch_df[patch_df["pid"] == pid]
        total = len(df_p)
        row = {"pid": pid,
               "surv": df_p["surv"].iloc[0],
               "status": df_p["status"].iloc[0]}
        feat_vec = []
        for c in sel_clusters:
            df_pc = df_p[df_p["cluster"] == c]
            n_c = len(df_pc)
            w_ij = n_c / total if total else 0.0
            if n_c:
                mat = torch.tensor(
                    df_pc.loc[:, [f"fea_{i}" for i in range(fe_dim)]].values,
                    dtype=torch.float32)
                x_bar = mat.mean(dim=0)
            else:
                x_bar = torch.zeros(fe_dim)
            weighted = (w_ij * x_bar).tolist()
            feat_vec.extend(weighted)
        for i, v in enumerate(feat_vec):
            row[f"fea_{i}"] = v
        out_rows.append(row)
    return pd.DataFrame(out_rows)

# ---------------- aggregation per fold ----------------
def aggregate_one_fold(fold, train_pids, valid_pids, test_pids):
    print(f"\n========== 开始第 {fold} 折聚合 (单折模式) ==========")
    splits = {"train": train_pids,
              "valid": valid_pids,
              "test":  test_pids}
    for split_name, pids in splits.items():
        print(f"\n--- 处理 {split_name} 集合: 共 {len(pids)} 位患者 ---")
        rows = patch_df[patch_df["pid"].isin(pids)].copy()
        print(f"    共载入 {len(rows)} 条 patch 记录")

        # 准备 DataFrames
        patch_feat_df = rows.reset_index(drop=True).copy()
        patch_risk_df = rows.reset_index(drop=True).copy()
        patch_risk_df["risk"] = 0.0

        # patch-level 推理
        with torch.no_grad():
            for c in tqdm(selected_cluster,
                          desc=f"[Fold{fold}-{split_name}] 簇推理",
                          leave=False,
                          ncols=80):
                mpath = MODEL_DIR / f"convimgmodel_cluster{c}_fold{fold}.pth"
                if not mpath.exists():
                    mpath = MODEL_DIR / f"convimgmodel_cluster{c}_fold1.pth"
                if not mpath.exists():
                    print(f"[Warn] 模型文件不存在: {mpath}")
                    continue

                model = DeepConvSurv(get_features=True)
                model.load_state_dict(torch.load(mpath,
                                                 map_location="cpu")["model"])
                model.eval()

                idxs = patch_feat_df.index[patch_feat_df["cluster"] == c].tolist()
                if not idxs:
                    print(f"    cluster {c}: 无 patch，跳过")
                    continue

                X, _ = make_patch_dataset(patch_feat_df.loc[idxs])
                if X is None:
                    print(f"    cluster {c}: X 为空，跳过")
                    continue

                feat, risk = model(X)
                feat_np = feat.cpu().numpy()
                risk_np = risk.cpu().numpy()
                D = feat_np.shape[1]
                cols = [f"fea_{i}" for i in range(D)]
                for col in cols:
                    if col not in patch_feat_df:
                        patch_feat_df[col] = 0.0
                patch_feat_df.loc[idxs, cols] = feat_np
                patch_risk_df.loc[idxs, "risk"] = risk_np
                print(f"    cluster {c}: 推理完成，patch 数={len(idxs)}")

        # patch → patient 加权
        fe_dim = D  # 从上面取出
        patient_fea  = agg_weighted_features(patch_feat_df,
                                             selected_cluster,
                                             fe_dim)
        patient_risk = agg_weighted_features(patch_risk_df,
                                             selected_cluster,
                                             fea_dim=1)
        print(f"    [Info] 聚合完成: 患者级特征维度={len(selected_cluster)*fe_dim}")

        # 保存 CSV
        out_fea = OUT_DIR / f"{split_name}_patient_features_fold{fold}.csv"
        out_rsk = OUT_DIR / f"{split_name}_patient_risks_fold{fold}.csv"
        patient_fea.to_csv(out_fea, index=False)
        patient_risk.to_csv(out_rsk, index=False)
        print(f"[Success] {split_name} 保存 → 特征:{out_fea.name}, 风险:{out_rsk.name}")

# ---------------- 单折执行 ----------------
all_pids = patient_df["pid"].tolist()
# 固定第一位为 test，第二位为 valid，其余为 train
test_pids  = [all_pids[0]]
valid_pids = [all_pids[1]]
train_pids = all_pids[2:]
print(f"\n>>> 单折配置信息:")
print(f"    test  患者: {test_pids}")
print(f"    valid 患者: {valid_pids}")
print(f"    train 患者: {train_pids}")

aggregate_one_fold(1, train_pids, valid_pids, test_pids)