#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cluster_select_holdout.py
------------------------
适配仅 **7** 位患者的 WSISA 实验管线：

1. 读取 `cluster_result/patches_1000_cls10_expanded.csv`
2. 循环每一个簇 c=0..9：
   * 过滤出 cluster==c 的 patch → df_c
   * 固定 1 人做测试、1 人做验证，其余 5 人训练
   * 调用 DeepConvSurv.fit() 训练 / 验证 / 测试
3. 打印并比较每个簇的 (val_C, test_C)
4. 给出验证 C‑index 最高的 **最佳簇**

整个脚本与目录无耦合，只需保证：
  WSISA/
  ├─ cluster_select_holdout.py
  ├─ DeepConvSurv_pytorch.py
  ├─ cluster_result/patches_1000_cls10_expanded.csv
  └─ data/patches/……
"""

import os, random, sys
import numpy as np
import pandas as pd

from DeepConvSurv_pytorch import DeepConvSurv

# ----------------------------------------------------------
BASE = os.path.abspath(os.path.dirname(__file__))
EXP_CSV = os.path.join(BASE, "cluster_result", "patches_1000_cls10_expanded.csv")

# 固定随机种子，确保可复现
SEED = 1
random.seed(SEED)
np.random.seed(SEED)
# ----------------------------------------------------------


def holdout_split(pids: np.ndarray):
    """
    7 个 pid → (train(5), val(1), test(1))
    随机打乱后取前 1 个做 test，后 1 个做 val，其余 train
    """
    assert len(pids) == 7, "只写死了 7 例留一方案，当前 pid 数目 ≠ 7"
    shuffled = pids.copy()
    np.random.shuffle(shuffled)
    test_pid, val_pid = shuffled[0], shuffled[1]
    train_pids = shuffled[2:]
    return train_pids, val_pid, test_pid


def run_one_cluster(df_c: pd.DataFrame, cluster_id: int):
    """在单簇内训练 / 评估 DeepConvSurv"""
    print(f"\n======  簇 {cluster_id}  ======")
    pids = df_c.pid.unique()
    if len(pids) < 3:
        print(f"[跳过] 该簇仅 {len(pids)} 位患者，无法做 5‑1‑1 划分")
        return None

    train_pids, val_pid, test_pid = holdout_split(pids)
    print("train:", train_pids)
    print("valid:", val_pid)
    print("test :", test_pid)

    # 将 pid → patch 行号
    def idx_of(pids_): return df_c.index[df_c.pid.isin(pids_)].tolist()
    tr_idx, va_idx, te_idx = idx_of(train_pids), idx_of([val_pid]), idx_of([test_pid])

    # 建模
    sample_img = df_c.patch_path.iloc[0]
    from PIL import Image
    im = Image.open(os.path.join(BASE, sample_img))
    c = len(im.getbands()); w, h = im.size

    net = DeepConvSurv(learning_rate=5e-4,
                       channel=c, width=w, height=h,
                       lr_decay=0.01)

    val_C, test_C = net.fit(df_c,
                            train_idx=tr_idx,
                            val_idx=va_idx,
                            test_idx=te_idx,
                            epochs=20,
                            batch_size=30,
                            verbose=False)
    print(f"[簇 {cluster_id}]  val C={val_C:.4f}  test C={test_C:.4f}")
    return val_C, test_C


def main():
    if not os.path.exists(EXP_CSV):
        print(f"[ERR] 找不到 {EXP_CSV}")
        sys.exit(1)

    df = pd.read_csv(EXP_CSV)
    print(f"读取扩展标签: {df.shape[0]} patch, {df.pid.nunique()} 患者, "
          f"{df.cluster.nunique()} 个簇")

    best_c, best_val = None, -1
    summary = []

    for c in sorted(df.cluster.unique()):
        df_c = df[df.cluster == c]
        result = run_one_cluster(df_c, c)
        if result is None:
            continue
        valC, testC = result
        summary.append((c, valC, testC))
        if valC > best_val:
            best_val, best_c = valC, c

    print("\n=======  各簇表现汇总  =======")
    for c, v, t in summary:
        print(f"簇 {c:<2d} | val C={v:.4f} | test C={t:.4f}")
    print("--------------------------------")
    print(f">>>  最佳簇 = {best_c}  (val C‑index={best_val:.4f})")


if __name__ == "__main__":
    main()