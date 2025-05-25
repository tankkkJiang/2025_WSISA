#!/usr/bin/env python
# expand_cluster_labels.py
# ------------------------------------------------------------
# 把聚类结果 CSV 与 TCGA 临床信息合并，生成带 surv / status 的补丁级标签
# ------------------------------------------------------------
import os
import pandas as pd
import numpy as np

# ----------------- 路径 -----------------
BASE_DIR      = os.path.abspath(os.path.dirname(__file__))
CLUSTER_CSV   = os.path.join(BASE_DIR, 'cluster_result', 'patches_1000_cls10.csv')
PATIENTS_CSV  = os.path.join(BASE_DIR, 'data', 'patients.csv')
OUT_CSV       = os.path.join(BASE_DIR, 'cluster_result', 'patches_1000_cls10_expanded.csv')

# ----------------- 1. 读聚类结果 -----------------
# 先尝试按有表头读取；若列数 < 4 再按无表头读取
try:
    df_cluster = pd.read_csv(CLUSTER_CSV)
    if df_cluster.shape[1] < 4 or 'pid' not in df_cluster.columns:
        raise ValueError
except Exception:
    df_cluster = pd.read_csv(
        CLUSTER_CSV,
        header=None,
        names=['patch_path', 'slide_id', 'pid', 'cluster']
    )

# ----------------- 2. 读患者表 -----------------
df_pat = pd.read_csv(PATIENTS_CSV, low_memory=False)

# 只保留我们需要的列：barcode, vital_status, days_to_death, days_to_last_follow_up
keep_cols = ['barcode', 'vital_status', 'days_to_death', 'days_to_last_follow_up']
df_pat = df_pat[keep_cols]

# （a）提取 pid = barcode 前 3 段，如 TCGA-BL-A3JM
df_pat['pid'] = df_pat['barcode'].apply(lambda x: '-'.join(str(x).split('-')[:3]))

# （b）生成 status（0=Alive, 1=Dead）
df_pat['status'] = df_pat['vital_status'].map({'Dead': 1, 'Alive': 0})

# （c）生存时间 surv
#    - 若死亡，用 days_to_death
#    - 否则，用 days_to_last_follow_up
df_pat['days_to_death'] = pd.to_numeric(df_pat['days_to_death'], errors='coerce')
df_pat['days_to_last_follow_up'] = pd.to_numeric(df_pat['days_to_last_follow_up'], errors='coerce')
df_pat['surv'] = np.where(
    df_pat['status'] == 1,
    df_pat['days_to_death'],
    df_pat['days_to_last_follow_up']
)

# 只保留 pid / surv / status
df_pat_short = df_pat[['pid', 'surv', 'status']]

# ----------------- 3. 合并 -----------------
df_expand = df_cluster.merge(df_pat_short, on='pid', how='left')

# ----------------- 4. 保存 -----------------
df_expand.to_csv(OUT_CSV, index=False)
print(f"[INFO] Saved expanded label to {OUT_CSV}")