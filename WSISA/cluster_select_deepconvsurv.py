#!/usr/bin/env python
# ------------------------------------------------------------
# cluster_select_deepconvsurv.py
# 1. 读取补丁级扩展标签 (含 surv/status)
# 2. 在病人级进行分层抽样 5‑fold
# 3. 调用 DeepConvSurv 训练各 fold
# ------------------------------------------------------------
import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

import DeepConvSurv_pytorch as deep_conv_surv  # 请确保已放在 PYTHONPATH

# ============= 全局路径 =============
BASE_DIR        = os.path.abspath(os.path.dirname(__file__))
DATA_DIR        = os.path.join(BASE_DIR, 'data')
PATCHES_ROOT    = os.path.join(DATA_DIR, 'patches')                # 所有 patch 根目录
EXP_LABEL_CSV   = os.path.join(BASE_DIR, 'cluster_result',
                               'patches_1000_cls10_expanded.csv')  # 补丁级标签 (已含 surv/status)
LOG_DIR         = os.path.join(BASE_DIR, 'log', 'wsisa_patch10')
os.makedirs(LOG_DIR, exist_ok=True)

# ============= 超参数 ==============
model_name = 'deepconvsurv'
epochs     = 20
lr         = 5e-4
seed       = 1
batchsize  = 30
train_test_ratio  = 0.9   # 病人级 train:test
train_valid_ratio = 0.9   # 训练集内部再划分 train:valid

# ------------------------------------------------------------
def convert_index(pid_list, expand_df):
    """
    将病人 pid 列表映射到 expand_df 的行索引
    返回 flat_index, patient_patch_counts
    """
    idx_per_patient = [expand_df.index[expand_df['pid'] == pid].tolist()
                       for pid in pid_list]
    counts = [len(x) for x in idx_per_patient]
    flat   = [i for sub in idx_per_patient for i in sub]
    return flat, counts


def main():
    # -------- 读补丁级扩展标签 --------
    expand_df = pd.read_csv(EXP_LABEL_CSV)
    if not {'pid', 'surv', 'status', 'cluster', 'patch_path'}.issubset(expand_df.columns):
        raise ValueError("扩展 CSV 缺少必要列，请检查生成步骤。")

    # -------- 病人级标签 --------
    labels_df = expand_df[['pid', 'surv', 'status']].drop_duplicates()
    labels_df = labels_df.reset_index(drop=True)

    # y = status (0/1)，用于 StratifiedKFold
    y_status = labels_df['status'].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    fold = 1
    test_c_indices = []   # 存放每 fold 的 test Concordance Index

    for train_pid_idx, test_pid_idx in skf.split(np.zeros(len(y_status)), y_status):
        print(f"\n================ Fold {fold} =================")
        # 病人 ID 划分
        train_pids = labels_df['pid'].iloc[train_pid_idx].values
        test_pids  = labels_df['pid'].iloc[test_pid_idx].values

        # 训练集再划分 train/valid
        sss = StratifiedShuffleSplit(n_splits=1,
                                     test_size=1 - train_valid_ratio,
                                     random_state=seed)
        train_inner_idx, valid_inner_idx = next(
            sss.split(np.zeros(len(train_pid_idx)), y_status[train_pid_idx])
        )
        train_inner_pids = train_pids[train_inner_idx]
        valid_inner_pids = train_pids[valid_inner_idx]

        # 映射到补丁级索引
        train_idx, _ = convert_index(train_inner_pids, expand_df)
        valid_idx, _ = convert_index(valid_inner_pids, expand_df)
        test_idx,  _ = convert_index(test_pids,        expand_df)

        # -------- 保存索引方便复现 --------
        np.savetxt(os.path.join(LOG_DIR, f"train_fold{fold}.csv"),
                   train_idx, delimiter=',', header='index', comments='')
        np.savetxt(os.path.join(LOG_DIR, f"valid_fold{fold}.csv"),
                   valid_idx, delimiter=',', header='index', comments='')
        np.savetxt(os.path.join(LOG_DIR, f"test_fold{fold}.csv"),
                   test_idx, delimiter=',',  header='index', comments='')

        # -------- DeepConvSurv 输入尺寸推断 --------
        sample_patch = Image.open(expand_df['patch_path'].iloc[0])
        w , h = sample_patch.size
        c    = len(sample_patch.getbands())

        # -------- 训练模型 --------
        hyper = dict(learning_rate=lr, channel=c, width=w, height=h)
        net   = deep_conv_surv.DeepConvSurv(**hyper)
        # 注意：network.train 内部需能处理 patch_path → 图像加载
        c_index = net.train(data_path=PATCHES_ROOT,
                            label_path=EXP_LABEL_CSV,
                            train_index=train_idx,
                            test_index=test_idx,
                            valid_index=valid_idx,
                            model_index=fold,
                            cluster=None,
                            batch_size=batchsize,
                            ratio=train_test_ratio,
                            num_epochs=epochs)
        test_c_indices.append(c_index)
        fold += 1

    print("\n================ Overall =================")
    print(f"C‑index mean: {np.mean(test_c_indices):.4f}  std: {np.std(test_c_indices):.4f}")


# ------------------------------------------------------------
if __name__ == "__main__":
    main()
else:
    print("cluster_select_deepconvsurv imported")