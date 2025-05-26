#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_WSISA_selectedCluster_7.py
 ‑ 单折聚合（显存友好版）
 ‑ 一次只加载一个簇的模型到 GPU，推理完立即释放
 ‑ 细粒度 tqdm + logging 便于跟踪
"""
import warnings, logging, time
from pathlib import Path

import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from networks import DeepConvSurv          # 保持原网络实现

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------- 日志 ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("wsisa_agg.log"),
              logging.StreamHandler()]
)

# ---------------- 路径与常量 ----------------
ROOT      = Path(__file__).resolve().parent
PATCH_CSV = ROOT / "cluster_result" / "patches_1000_cls10_expanded.csv"
MODEL_DIR = ROOT / "log" / "wsisa_patch10" / "convimgmodel"
SEL_FILE  = ROOT / "log" / "selected_clusters.txt"
OUT_DIR   = ROOT / "results"; OUT_DIR.mkdir(exist_ok=True, parents=True)

MEAN = [0.6964, 0.5905, 0.6692]
STD  = [0.2559, 0.2943, 0.2462]
GPU_BATCH   = 16               # —— 单次送进 GPU 的 patch 数
CPU_WORKERS = 4
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.info("===== 运行配置 =====")
for k, v in dict(PATCH_CSV=PATCH_CSV,
                 MODEL_DIR=MODEL_DIR,
                 SEL_FILE=SEL_FILE,
                 OUT_DIR=OUT_DIR,
                 GPU_BATCH=GPU_BATCH,
                 DEVICE=str(DEVICE)).items():
    logging.info(f"{k:<10}: {v}")

# ---------------- 读取 CSV ----------------
selected_cluster = [int(x) for x in SEL_FILE.read_text().split(",") if x.strip().isdigit()]
logging.info(f"已选簇 ({len(selected_cluster)}): {selected_cluster}")

patch_df   = pd.read_csv(PATCH_CSV,
                         usecols=["pid", "surv", "status", "patch_path", "cluster"])
patient_df = patch_df[["pid", "surv", "status"]].drop_duplicates()
logging.info(f"患者数: {len(patient_df)}  |  Patch 行数: {len(patch_df)}")

# ---------------- 数据集 ----------------
TRANSF = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])

class SimpleDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(ROOT / self.paths[idx]).convert("RGB")
        return TRANSF(img)

# ---------------- 主折运行 ----------------
def run_fold(fold_id: int, train_p, valid_p, test_p):
    SPLIT = dict(train=train_p, valid=valid_p, test=test_p)

    for split, pids in SPLIT.items():
        logging.info(f"=== [{split.upper()}] 共有 {len(pids)} 位患者 ===")
        df_split = patch_df[patch_df["pid"].isin(pids)].reset_index(drop=True)

        # 先占位大矩阵
        feats = None        # numpy array，稍后初始化 (N, feat_dim)
        risks = np.zeros(len(df_split), dtype=np.float32)

        t_start_split = time.time()

        # —— 逐簇处理：一次只让一个模型上 GPU ——
        for c in tqdm(selected_cluster, desc=f"[{split}] clusters", ncols=80):
            idxs = np.where(df_split["cluster"].values == c)[0]
            if len(idxs) == 0:
                continue

            paths_c = df_split.loc[idxs, "patch_path"].tolist()
            loader  = DataLoader(SimpleDataset(paths_c),
                                 batch_size=GPU_BATCH,
                                 shuffle=False,
                                 num_workers=CPU_WORKERS,
                                 pin_memory=True)

            # ---- 加载模型（仅 CPU） ----
            mpath = MODEL_DIR / f"convimgmodel_cluster{c}_fold{fold_id}.pth"
            if not mpath.exists():
                mpath = MODEL_DIR / f"convimgmodel_cluster{c}_fold1.pth"
            model = DeepConvSurv(get_features=True)
            model.load_state_dict(torch.load(mpath, map_location="cpu")["model"])
            model.eval().to(DEVICE)

            feats_c = []
            risks_c = []

            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(DEVICE, non_blocking=True)
                    f, r  = model(batch)
                    feats_c.append(f.cpu())
                    risks_c.append(r.cpu())

            # ---- 回收显存 ----
            model.cpu(); del model
            torch.cuda.empty_cache()

            # ---- 写回主数组 ----
            feats_c = torch.cat(feats_c).numpy()
            risks_c = torch.cat(risks_c).numpy()

            if feats is None:
                feat_dim = feats_c.shape[1]
                feats    = np.zeros((len(df_split), feat_dim), dtype=np.float32)

            feats[idxs] = feats_c
            risks[idxs] = risks_c

            tqdm.write(f"簇 {c:<2} 推理完成  patch={len(idxs)}")

        logging.info(f"[{split}] All clusters done,  耗时 {time.time()-t_start_split:.1f}s")

        # ---------- patch → patient 加权 ----------
        feat_cols = [f"fea_{i}" for i in range(feats.shape[1])]
        df_split[feat_cols] = feats
        df_split["risk"]    = risks

        def agg_patient(group, field, dim):
            total = len(group)
            out   = np.zeros(len(selected_cluster)*dim, dtype=np.float32)
            for j, c in enumerate(selected_cluster):
                g_c = group[group["cluster"] == c]
                w   = len(g_c) / total if total else 0
                if len(g_c):
                    vec = g_c[field].values.mean(axis=0)
                else:
                    vec = np.zeros(dim, dtype=np.float32)
                out[j*dim:(j+1)*dim] = w * vec
            return out

        patient_rows = []
        for pid, g in df_split.groupby("pid"):
            fea_vec = agg_patient(g, feat_cols, feats.shape[1])
            risk_v  = agg_patient(g, ["risk"], 1)[0]
            patient_rows.append(dict(pid=pid,
                                     surv=int(g["surv"].iat[0]),
                                     status=int(g["status"].iat[0]),
                                     **{f"fea_{i}": v for i, v in enumerate(fea_vec)},
                                     risk=risk_v))
        pat_df = pd.DataFrame(patient_rows)

        # ---------- 保存 ----------
        fea_csv = OUT_DIR / f"{split}_patient_features_fold{fold_id}.csv"
        rsk_csv = OUT_DIR / f"{split}_patient_risks_fold{fold_id}.csv"
        pat_df.drop(columns="risk").to_csv(fea_csv, index=False)
        pat_df[["pid", "risk"]].to_csv(rsk_csv, index=False)
        logging.info(f"[✓] {split} 保存完毕  →  {fea_csv.name} / {rsk_csv.name}")

# ------------------ 折划分 ------------------
all_pids = patient_df["pid"].tolist()
test_p, valid_p, train_p = [all_pids[0]], [all_pids[1]], all_pids[2:]
logging.info(f"Fold‑1: test={test_p}  valid={valid_p}  train={train_p}")

run_fold(1, train_p, valid_p, test_p)