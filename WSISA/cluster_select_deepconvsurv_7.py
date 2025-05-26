#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_WSISA_selectedCluster_fast.py     ——  单折聚合 · 精简版
 ‑ 读取已选簇的 patch‑level 模型
 ‑ DataLoader 批量推理  →  patch  →  patient 加权平均
 ‑ 保存 train / valid / test 患者级特征 & 风险
"""
import warnings, random
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from networks import DeepConvSurv   # 与旧脚本相同

warnings.filterwarnings("ignore", category=FutureWarning)

# --------- 路径与配置 ---------
ROOT      = Path(__file__).resolve().parent
PATCH_CSV = ROOT / "cluster_result" / "patches_1000_cls10_expanded.csv"
MODEL_DIR = ROOT / "log" / "wsisa_patch10" / "convimgmodel"
SEL_FILE  = ROOT / "log" / "selected_clusters.txt"
OUT_DIR   = ROOT / "results"; OUT_DIR.mkdir(exist_ok=True, parents=True)

MEAN = [0.6964, 0.5905, 0.6692]
STD  = [0.2559, 0.2943, 0.2462]
BATCH_SIZE   = 64          # ← 可自行调大；显存不足时调小
NUM_WORKERS  = 4           # ← cpu 线程
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(">>> 运行配置")
for k, v in dict(PATCH_CSV=PATCH_CSV,
                 MODEL_DIR=MODEL_DIR,
                 SEL_FILE=SEL_FILE,
                 OUT_DIR=OUT_DIR,
                 BATCH_SIZE=BATCH_SIZE,
                 DEVICE=str(DEVICE)).items():
    print(f"  {k:<10}: {v}")

# ---------- 读取数据 ----------
selected_cluster = [int(x) for x in SEL_FILE.read_text().split(",") if x.strip().isdigit()]
print(f">>> 已选簇 ({len(selected_cluster)}): {selected_cluster}")

patch_df = pd.read_csv(PATCH_CSV, usecols=["pid","surv","status","patch_path","cluster"])
patient_df = patch_df[["pid","surv","status"]].drop_duplicates()
print(f">>> 患者数: {len(patient_df)} / Patch 行数: {len(patch_df)}")

# ---------- Dataset ----------
TRANSF = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])

class PatchDataset(Dataset):
    def __init__(self, rows: pd.DataFrame):
        self.paths   = rows["patch_path"].tolist()
        self.clu_ids = rows["cluster"].tolist()
        self.pids    = rows["pid"].tolist()
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(ROOT / self.paths[idx]).convert("RGB")
        return TRANSF(img), self.clu_ids[idx], self.pids[idx]

# ---------- 主流程 ----------
def run_single_fold(fold_id: int, train_p, valid_p, test_p):
    SPLIT = dict(train=train_p, valid=valid_p, test=test_p)
    for split, pids in SPLIT.items():
        print(f"\n=== {split.upper()}  ({len(pids)} patients) ===")
        df_split = patch_df[patch_df["pid"].isin(pids)].reset_index(drop=True)
        loader   = DataLoader(PatchDataset(df_split),
                              batch_size=BATCH_SIZE,
                              shuffle=False,
                              num_workers=NUM_WORKERS,
                              pin_memory=True)

        # 预创建结果数组
        fea_dim = None
        feats   = torch.zeros(len(df_split), 1)   # 临时占位，稍后扩展
        risks   = torch.zeros(len(df_split))

        # 模型缓存 —— 同一个簇只 load 一次
        model_cache: dict[int, torch.nn.Module] = {}

        with torch.no_grad():
            start = 0
            for imgs, clu_ids, _ in tqdm(loader, desc=f"[{split}] 推理", ncols=80):
                batch = imgs.to(DEVICE)

                # 按簇分批推理，避免一次 forward N 个模型
                for c in torch.unique(clu_ids):
                    idx = (clu_ids == c).nonzero(as_tuple=True)[0]
                    sub = batch[idx]

                    if c.item() not in model_cache:
                        mpath = (MODEL_DIR / f"convimgmodel_cluster{c}_fold{fold_id}.pth")
                        if not mpath.exists():
                            mpath = (MODEL_DIR / f"convimgmodel_cluster{c}_fold1.pth")
                        model           = DeepConvSurv(get_features=True)
                        model.load_state_dict(torch.load(mpath, map_location=DEVICE)["model"])
                        model.eval().to(DEVICE)
                        model_cache[c.item()] = model

                    feat, risk = model_cache[c.item()](sub)
                    if fea_dim is None:
                        fea_dim = feat.shape[1]
                        feats   = torch.zeros(len(df_split), fea_dim, device="cpu")

                    feats[start+idx] = feat.cpu()
                    risks[start+idx] = risk.cpu()

                start += len(imgs)

        # 附回 DataFrame
        df_split[[f"fea_{i}" for i in range(fea_dim)]] = feats.numpy()
        df_split["risk"] = risks.numpy()

        # ------ patch→patient 加权 ------
        #   直接利用 groupby 减少手写循环
        def _agg(group, field_mat, dim):
            total = len(group)
            out   = torch.zeros(len(selected_cluster)*dim)
            for j, c in enumerate(selected_cluster):
                sub = group[group["cluster"] == c]
                w   = len(sub)/total if total else 0
                if len(sub):
                    vec = torch.tensor(sub[field_mat].values).float().mean(0)
                else:
                    vec = torch.zeros(dim)
                out[j*dim:(j+1)*dim] = w*vec
            return out.tolist()

        patient_rows = []
        fea_cols = [f"fea_{i}" for i in range(fea_dim)]
        for pid, g in df_split.groupby("pid"):
            vec_fea = _agg(g, fea_cols, fea_dim)
            vec_rsk = _agg(g, ["risk"], 1)
            patient_rows.append(dict(pid=pid,
                                     surv=g["surv"].iat[0],
                                     status=g["status"].iat[0],
                                     **{f"fea_{i}":v for i,v in enumerate(vec_fea)},
                                     risk=vec_rsk[0]))
        pat_df = pd.DataFrame(patient_rows)

        # -------- 保存 --------
        fea_csv = OUT_DIR / f"{split}_patient_features_fold{fold_id}.csv"
        rsk_csv = OUT_DIR / f"{split}_patient_risks_fold{fold_id}.csv"
        pat_df.drop(columns="risk").to_csv(fea_csv, index=False)
        pat_df[["pid","risk"]].to_csv(rsk_csv, index=False)
        print(f"[✓] 保存 {fea_csv.name} / {rsk_csv.name}")

# ---------- 单折划分 ----------
all_pids = patient_df["pid"].tolist()
test_p, valid_p, train_p = [all_pids[0]], [all_pids[1]], all_pids[2:]
print(f"\n>>> Fold‑1 划分  test={test_p} valid={valid_p} train={train_p}")

run_single_fold(1, train_p, valid_p, test_p)