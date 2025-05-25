#!/usr/bin/env python
import os
import pandas as pd

BASE_DIR     = os.path.abspath(os.path.dirname(__file__))
CLUSTER_CSV  = os.path.join(BASE_DIR, 'cluster_result', 'patches_1000_cls10.csv')
PATIENTS_CSV = os.path.join(BASE_DIR, 'data', 'patients.csv')

# 读入
df_cluster  = pd.read_csv(CLUSTER_CSV, header=None, names=['patch_path','pid','cluster'])
df_patient  = pd.read_csv(PATIENTS_CSV)  # 列 ['pid','surv','status']

# 合并
df_expand = df_cluster.merge(df_patient, on='pid', how='left')

# 保存到新的文件
OUT = os.path.join(BASE_DIR, 'cluster_result', 'patches_1000_cls10_expanded.csv')
df_expand.to_csv(OUT, index=False)

print(f"Saved expanded label to {OUT}")