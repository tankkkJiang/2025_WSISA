#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_WSISA_selectedCluster_safe_stream.py
  • 单折聚合（GPU/CPU 内存极限省）
  • batch 结果即刻累加，不保留任何 patch‑level列表
  • 一次只让一个模型驻留 GPU；推完即释放
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

# ----------- 环境 & 日志 -----------
warnings.filterwarnings("ignore", category=FutureWarning)
torch.set_grad_enabled(False)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("wsisa_agg_stream.log"),
              logging.StreamHandler()]
)

# ----------- 路径 & 全局配置 -----------
ROOT       = Path(__file__).resolve().parent
PATCH_CSV  = ROOT / "cluster_result" / "patches_1000_cls10_expanded.csv"
MODEL_DIR  = ROOT / "log" / "wsisa_patch10" / "convimgmodel"
SEL_FILE   = ROOT / "log" / "selected_clusters.txt"
OUT_DIR    = ROOT / "results"; OUT_DIR.mkdir(exist_ok=True, parents=True)

MEAN, STD   = [0.6964,0.5905,0.6692], [0.2559,0.2943,0.2462]
GPU_BATCH   = 8            # 进一步缩小 batch
CPU_WORKERS = 0            # 主进程加载 —— 最省内存
PIN_MEMORY  = False
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for k,v in dict(GPU_BATCH=GPU_BATCH, DEVICE=str(DEVICE)).items():
    logging.info(f"{k}: {v}")

# ----------- 数据准备 -----------
selected_cluster = [int(x) for x in SEL_FILE.read_text().split(",") if x.strip().isdigit()]
logging.info(f"已选簇: {selected_cluster}")

patch_df   = pd.read_csv(PATCH_CSV, usecols=["pid","surv","status","patch_path","cluster"])
patient_df = patch_df[["pid","surv","status"]].drop_duplicates()
logging.info(f"患者数: {len(patient_df)}  |  Patch 数: {len(patch_df)}")

# ----------- Dataset -----------
TRANSF = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])

class ClusterDataset(Dataset):
    def __init__(self, paths, glb_idxs):  # glb_idxs 用于快速取 pid
        self.paths, self.gidx = paths, glb_idxs
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        img_path = ROOT / self.paths[i]
        img = Image.open(img_path).convert("RGB")
        tensor = TRANSF(img)
        img.close()
        return tensor, self.gidx[i]

# ----------- 主流程 -----------
def run_fold(fold_id:int, train_p, valid_p, test_p):
    LOG = logging.info
    for split,pids in dict(train=train_p, valid=valid_p, test=test_p).items():
        LOG(f"=== [{split.upper()}] 患者 {pids}")
        df_split = patch_df[patch_df["pid"].isin(pids)].reset_index(drop=True)
        total_cnt = Counter(df_split["pid"])

        # 累计容器  (cluster -> pid -> 累加向量 / 累加风险)
        sum_feat = {c: defaultdict(lambda: None) for c in selected_cluster}
        sum_risk = {c: defaultdict(float)       for c in selected_cluster}
        feat_dim = None

        t0 = time.time()
        for c in tqdm(selected_cluster, desc=f"[{split}] cluster", ncols=80):
            idxs = np.where(df_split["cluster"].values == c)[0]
            if not len(idxs):
                continue

            ds = ClusterDataset(df_split.loc[idxs,"patch_path"].tolist(), idxs.tolist())
            loader = DataLoader(ds, batch_size=GPU_BATCH, shuffle=False,
                                num_workers=CPU_WORKERS, pin_memory=PIN_MEMORY)

            # load model to DEVICE
            mdl_path = MODEL_DIR / f"convimgmodel_cluster{c}_fold{fold_id}.pth"
            if not mdl_path.exists():
                mdl_path = MODEL_DIR / f"convimgmodel_cluster{c}_fold1.pth"
            model = DeepConvSurv(get_features=True)
            model.load_state_dict(torch.load(mdl_path, map_location="cpu")["model"])
            model.to(DEVICE).eval()

            # ----- 批推理 & 立即累加 -----
            for imgs, gidx in loader:
                imgs = imgs.to(DEVICE, non_blocking=True)
                fts, rks = model(imgs)
                fts = fts.cpu().numpy(); rks = rks.cpu().numpy()

                if feat_dim is None:
                    feat_dim = fts.shape[1]

                for j, glb in enumerate(gidx):
                    pid = df_split.at[int(glb), "pid"]
                    if sum_feat[c][pid] is None:
                        sum_feat[c][pid] = np.zeros(feat_dim, dtype=np.float32)
                    sum_feat[c][pid] += fts[j]
                    sum_risk[c][pid] += float(rks[j])

            # 释放
            model.cpu(); del model
            torch.cuda.empty_cache()
            tqdm.write(f"cluster {c} OK  patches={len(idxs)}")

        LOG(f"[{split}] 推理完成，用时 {time.time()-t0:.1f}s")

        # ---------- 生成患者级特征 ----------
        rows=[]
        for pid in pids:
            surv   = int(df_split[df_split["pid"]==pid]["surv"].iat[0])
            status = int(df_split[df_split["pid"]==pid]["status"].iat[0])
            tot    = total_cnt[pid]

            fea_vec=[]; risk_vec=[]
            for c in selected_cluster:
                vec = sum_feat[c].get(pid)
                if vec is None:
                    fea_block = np.zeros(feat_dim, dtype=np.float32)
                    rk_block  = 0.0
                else:
                    fea_block = vec / tot
                    rk_block  = sum_risk[c][pid] / tot
                fea_vec.extend(fea_block.tolist())
                risk_vec.append(rk_block)

            rows.append({"pid":pid,"surv":surv,"status":status,
                         **{f"fea_{i}":v for i,v in enumerate(fea_vec)},
                         **{f"risk_{j}":v for j,v in enumerate(risk_vec)}})

        pat_df = pd.DataFrame(rows)

        # ---------- 保存 ----------
        feat_cols=[f"fea_{i}" for i in range(len(selected_cluster)*feat_dim)]
        risk_cols=[f"risk_{j}" for j in range(len(selected_cluster))]
        pat_df[["pid","surv","status"]+feat_cols]\
              .to_csv(OUT_DIR/f"{split}_patient_features_fold{fold_id}.csv", index=False)
        pat_df[["pid","surv","status"]+risk_cols]\
              .to_csv(OUT_DIR/f"{split}_patient_risks_fold{fold_id}.csv", index=False)
        LOG(f"[✓] {split} 保存完毕")

# ----------- 折划分 & 执行 -----------
all_pids=patient_df["pid"].tolist()
run_fold(1, train_p=all_pids[2:], valid_p=[all_pids[1]], test_p=[all_pids[0]])