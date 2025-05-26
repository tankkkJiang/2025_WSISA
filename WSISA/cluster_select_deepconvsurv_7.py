#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_WSISA_selectedCluster_safe_mem.py
 - 单折聚合（显存 + CPU 内存友好版）
 - 在线累积：不在内存中保存所有 patch-level 特征
 - 一次只加载一个簇的模型到 GPU，推理完立即释放
 - 细粒度 tqdm + logging 便于跟踪
"""
import warnings, logging, time
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from networks import DeepConvSurv

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------- 日志 ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("wsisa_agg_mem.log"),
              logging.StreamHandler()]
)

# ---------------- 路径 & 配置 ----------------
ROOT      = Path(__file__).resolve().parent
PATCH_CSV = ROOT / "cluster_result" / "patches_1000_cls10_expanded.csv"
MODEL_DIR = ROOT / "log" / "wsisa_patch10" / "convimgmodel"
SEL_FILE  = ROOT / "log" / "selected_clusters.txt"
OUT_DIR   = ROOT / "results"; OUT_DIR.mkdir(exist_ok=True, parents=True)

MEAN       = [0.6964, 0.5905, 0.6692]
STD        = [0.2559, 0.2943, 0.2462]
GPU_BATCH  = 16         # 一次送进 GPU 的 patch 数，显存太小可再调小
CPU_WORKERS= 0          # DataLoader 进程数，0 表示主进程加载，避免内存泄漏
PIN_MEMORY = False      # False 可减少 CPU 端内存压力
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.info("===== 运行配置 =====")
for k, v in dict(PATCH_CSV=PATCH_CSV,
                 MODEL_DIR=MODEL_DIR,
                 SEL_FILE=SEL_FILE,
                 OUT_DIR=OUT_DIR,
                 GPU_BATCH=GPU_BATCH,
                 CPU_WORKERS=CPU_WORKERS,
                 PIN_MEMORY=PIN_MEMORY,
                 DEVICE=str(DEVICE)).items():
    logging.info(f"{k:<12}: {v}")

# ---------------- 读取 CSV ----------------
selected_cluster = [int(x) for x in SEL_FILE.read_text().split(",") if x.strip().isdigit()]
logging.info(f"已选簇 ({len(selected_cluster)}): {selected_cluster}")

patch_df   = pd.read_csv(PATCH_CSV, usecols=["pid","surv","status","patch_path","cluster"])
patient_df = patch_df[["pid","surv","status"]].drop_duplicates()
logging.info(f"患者数: {len(patient_df)} | Patch 总行数: {len(patch_df)}")

# ---------------- 数据集 ----------------
TRANSF = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])

class ClusterDataset(Dataset):
    """只返回图像 tensor 和它在 df_split 中的全局索引"""
    def __init__(self, paths, glb_indices):
        self.paths       = paths
        self.glb_indices = glb_indices
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, i):
        img = Image.open(ROOT / self.paths[i]).convert("RGB")
        return TRANSF(img), self.glb_indices[i]

# ---------------- 主流程 ----------------
def run_fold(fold_id: int, train_p, valid_p, test_p):
    LOG = logging.info
    for split_name, pids in dict(train=train_p, valid=valid_p, test=test_p).items():
        LOG(f"=== [{split_name.upper()}] {len(pids)} 名患者 ===")
        df_split = patch_df[patch_df["pid"].isin(pids)].reset_index(drop=True)
        total_counts = Counter(df_split["pid"].tolist())

        # 在线累积容器
        sum_feats  = {c: defaultdict(lambda: None) for c in selected_cluster}
        sum_risks  = {c: defaultdict(float)     for c in selected_cluster}

        t0_split = time.time()

        # —— 逐簇推理 & 在线累积 ——
        for c in tqdm(selected_cluster, desc=f"[{split_name}] clusters", ncols=80):
            idxs = np.where(df_split["cluster"].values == c)[0]
            if len(idxs) == 0:
                continue

            paths_c = df_split.loc[idxs, "patch_path"].tolist()
            ds      = ClusterDataset(paths_c, idxs.tolist())
            loader  = DataLoader(ds,
                                 batch_size=GPU_BATCH,
                                 shuffle=False,
                                 num_workers=CPU_WORKERS,
                                 pin_memory=PIN_MEMORY)

            # 加载模型到 DEVICE
            mpath = MODEL_DIR / f"convimgmodel_cluster{c}_fold{fold_id}.pth"
            if not mpath.exists():
                mpath = MODEL_DIR / f"convimgmodel_cluster{c}_fold1.pth"
            model = DeepConvSurv(get_features=True)
            model.load_state_dict(torch.load(mpath, map_location="cpu")["model"])
            model.to(DEVICE).eval()

            # 推理并在线累积
            for imgs, glb_idxs in tqdm(loader,
                                       desc=f"  cluster {c:>2} batches",
                                       leave=False,
                                       ncols=80):
                imgs = imgs.to(DEVICE, non_blocking=True)
                with torch.no_grad():
                    feats, risks = model(imgs)
                feats = feats.cpu().numpy()
                risks = risks.cpu().numpy()

                for j, glb in enumerate(glb_idxs):
                    pid = df_split.at[glb, "pid"]
                    # 初始化 sum_feats[c][pid]
                    if sum_feats[c][pid] is None:
                        sum_feats[c][pid] = np.zeros(feats.shape[1], dtype=np.float32)
                    sum_feats[c][pid] += feats[j]
                    sum_risks[c][pid] += float(risks[j])

            # 回收显存
            model.cpu()
            del model
            torch.cuda.empty_cache()

            tqdm.write(f"簇 {c:<2} 完成，patch={len(idxs)}")

        LOG(f"[{split_name}] all clusters done in {time.time()-t0_split:.1f}s")

        # ---------- 构建患者级 DataFrame ----------
        feat_dim = next(iter(sum_feats.values()))[next(iter(sum_feats[selected_cluster[0]].keys()))].shape[0]
        rows = []
        for pid in pids:
            surv   = int(df_split[df_split["pid"] == pid]["surv"].iat[0])
            status = int(df_split[df_split["pid"] == pid]["status"].iat[0])
            total  = total_counts[pid]
            fea_vec = []
            rsk_vec = []
            for c in selected_cluster:
                sf = sum_feats[c].get(pid)
                if sf is None:
                    fea_block = np.zeros(feat_dim, dtype=np.float32)
                    rsk_block = 0.0
                else:
                    # weighted mean = sum_feats[c][pid] / total_counts[pid]
                    fea_block = sf / total
                    rsk_block = sum_risks[c][pid] / total
                fea_vec.extend(fea_block.tolist())
                rsk_vec.append(rsk_block)
            row = {
                "pid": pid,
                "surv": surv,
                "status": status,
                **{f"fea_{i}": v for i, v in enumerate(fea_vec)},
                **{f"risk_{j}": v for j, v in enumerate(rsk_vec)}
            }
            rows.append(row)
        out_df = pd.DataFrame(rows)

        # ---------- 保存 CSV ----------
        fea_cols = [f"fea_{i}" for i in range(len(selected_cluster)*feat_dim)]
        risk_cols= [f"risk_{j}" for j in range(len(selected_cluster))]
        fea_csv = OUT_DIR / f"{split_name}_patient_features_fold{fold_id}.csv"
        rsk_csv = OUT_DIR / f"{split_name}_patient_risks_fold{fold_id}.csv"

        out_df[["pid","surv","status"] + fea_cols].to_csv(fea_csv, index=False)
        out_df[["pid","surv","status"] + risk_cols].to_csv(rsk_csv, index=False)
        LOG(f"[✓] [{split_name}] 保存 → 特征:{fea_csv.name}, 风险:{rsk_csv.name}")

# ---------------- 折划分 & 执行 ----------------
all_pids = patient_df["pid"].tolist()
test_p  = [all_pids[0]]
valid_p = [all_pids[1]]
train_p = all_pids[2:]
logging.info(f"Fold-1  test={test_p}  valid={valid_p}  train={train_p}")

run_fold(1, train_p, valid_p, test_p)