#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_WSISA_selectedCluster.py
  • 单折聚合（GPU／CPU 内存友好 · 支持多张 WSI）
  • 一次只加载一个 model 到 GPU，推完即释放
  • 超过 MAX_PATCHES 的簇跳过并发出警告
  • LOPO over slide_id（每张 WSI 都做一次留一法）
"""
import warnings, logging, time, random
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from networks import DeepConvSurv

# -------- 环境 & 日志 --------
warnings.filterwarnings("ignore", category=FutureWarning)
torch.set_grad_enabled(False)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler("wsisa_gpu_accum_skip_multiWSI.log"),
              logging.StreamHandler()]
)

# -------- 全局配置 --------
ROOT        = Path(__file__).resolve().parent
PATCH_CSV   = ROOT / "cluster_result" / "patches_1000_cls10_expanded.csv"
MODEL_DIR   = ROOT / "log" / "wsisa_patch10" / "convimgmodel"
SEL_FILE    = ROOT / "log" / "selected_clusters.txt"
OUT_DIR     = ROOT / "results"; OUT_DIR.mkdir(exist_ok=True, parents=True)

MEAN, STD   = [0.6964,0.5905,0.6692], [0.2559,0.2943,0.2462]
GPU_BATCH   = 8       # 每次送 GPU 的 patch 数
CPU_WORKERS = 0       # DataLoader 线程数
PIN_MEMORY  = False
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_PATCHES = 7000    # 超过此 patch 数的簇跳过

logging.info(f"DEVICE={DEVICE}, GPU_BATCH={GPU_BATCH}, MAX_PATCHES={MAX_PATCHES}")

# -------- 读取 & 预处理 --------
patch_df = pd.read_csv(PATCH_CSV,
                       usecols=["slide_id","pid","surv","status","patch_path","cluster"])
# 用 slide_id 作为 LOPO 单位
wsi_list = patch_df["slide_id"].unique().tolist()
logging.info(f"共发现 {len(wsi_list)} 张 WSI")

# 读取已选簇
selected_cluster = [int(x) for x in SEL_FILE.read_text().split(",") if x.strip().isdigit()]
logging.info(f"选簇: {selected_cluster}")

# -------- Dataset & DataLoader --------
TRANSF = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
class ClusterDataset(Dataset):
    def __init__(self, paths, glb_idxs):
        self.paths    = paths
        self.glb_idxs = glb_idxs
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, i):
        img = Image.open(ROOT / self.paths[i]).convert("RGB")
        x   = TRANSF(img)
        img.close()
        return x, self.glb_idxs[i]

def make_loader(paths, glb_idxs):
    ds = ClusterDataset(paths, glb_idxs)
    return DataLoader(ds,
                      batch_size=GPU_BATCH,
                      shuffle=False,
                      num_workers=CPU_WORKERS,
                      pin_memory=PIN_MEMORY,
                      drop_last=False)

# -------- 核心聚合流程 --------
def run_LOPO():
    for fold_id, test_wsi in enumerate(wsi_list, start=1):
        # 划分 train/valid/test by slide_id
        remaining = [w for w in wsi_list if w != test_wsi]
        random.shuffle(remaining)
        valid_wsi = [remaining.pop()]  # 随机一张做验证
        train_wsi = remaining

        logging.info(f"\n=== Fold {fold_id}: test={test_wsi}, valid={valid_wsi}, train={len(train_wsi)} WSIs ===")
        for split, wsi_sub in {"train":train_wsi, "valid":valid_wsi, "test":[test_wsi]}.items():
            logging.info(f"--- [{split}] 共 {len(wsi_sub)} 张 WSI ---")
            df_sub = patch_df[patch_df["slide_id"].isin(wsi_sub)].reset_index(drop=True)
            total_cnt = Counter(df_sub["slide_id"])
            wsi2i    = {w: i for i,w in enumerate(wsi_sub)}
            nC, nW   = len(selected_cluster), len(wsi_sub)

            sum_feat = None   # torch.zeros(nC, nW, feat_dim, device=DEVICE)
            sum_risk = None   # torch.zeros(nC, nW, device=DEVICE)

            t0 = time.time()
            for c_idx, c in enumerate(tqdm(selected_cluster,
                                          desc=f"[{split}] clusters",
                                          ncols=80)):
                idxs = np.where(df_sub["cluster"].values == c)[0]
                n_patches = len(idxs)
                if n_patches == 0:
                    continue
                if n_patches > MAX_PATCHES:
                    logging.warning(f"簇 {c} patch={n_patches} 超过 {MAX_PATCHES}, 跳过")
                    continue

                paths = df_sub.loc[idxs, "patch_path"].tolist()
                loader= make_loader(paths, idxs.tolist())

                # 只加载对应簇模型
                mpath = MODEL_DIR / f"convimgmodel_cluster{c}_fold{fold_id}.pth"
                if not mpath.exists():
                    mpath = MODEL_DIR / f"convimgmodel_cluster{c}_fold1.pth"
                model = DeepConvSurv(get_features=True)
                model.load_state_dict(torch.load(mpath, map_location="cpu")["model"])
                model.to(DEVICE).eval()

                for imgs, glb_idxs in loader:
                    imgs = imgs.to(DEVICE, non_blocking=True)
                    fts, rks = model(imgs)  # fts:(B,D), rks:(B,1) or (B,)
                    # init accumulators
                    if sum_feat is None:
                        D = fts.shape[1]
                        sum_feat = torch.zeros(nC, nW, D, device=DEVICE)
                        sum_risk = torch.zeros(nC, nW,   device=DEVICE)
                    # 分配到各 WSI 索引并累加
                    pidxs = torch.tensor([ wsi2i[df_sub.at[int(g),"slide_id"]]
                                           for g in glb_idxs ], device=DEVICE)
                    for wi in torch.unique(pidxs):
                        mask = (pidxs == wi)
                        sum_feat[c_idx, wi] += fts[mask].sum(dim=0)
                        sum_risk[c_idx, wi] += rks[mask].view(-1).sum()

                # 卸载模型
                model.cpu();  del model;  torch.cuda.empty_cache()
                logging.info(f"簇 {c} 推理完成, patch={n_patches}")

            logging.info(f"[{split}] 推理耗时 {time.time()-t0:.1f}s")

            # 拉回 CPU，做平均并保存
            sf = sum_feat.cpu().numpy()  # (C, W, D)
            sr = sum_risk.cpu().numpy()  # (C, W)

            rows = []
            for wi, wsi in enumerate(wsi_sub):
                surv   = int(df_sub[df_sub["slide_id"]==wsi]["surv"].iat[0])
                status = int(df_sub[df_sub["slide_id"]==wsi]["status"].iat[0])
                tot    = total_cnt[wsi]
                fea, rsk = [], []
                for ci in range(nC):
                    blk = sf[ci, wi]
                    fea.extend((blk/tot).tolist())
                    rsk.append((sr[ci, wi]/tot).item())
                row = {
                    "slide_id": wsi, "surv": surv, "status": status,
                    **{f"fea_{i}":v for i,v in enumerate(fea)},
                    **{f"risk_{j}":v for j,v in enumerate(rsk)}
                }
                rows.append(row)

            out = pd.DataFrame(rows)
            feat_cols = [f"fea_{i}"  for i in range(nC * D)]
            risk_cols = [f"risk_{j}" for j in range(nC)]
            out[["slide_id","surv","status"]+feat_cols]\
                .to_csv(OUT_DIR/f"{split}_WSI_features_fold{fold_id}.csv", index=False)
            out[["slide_id","surv","status"]+risk_cols]\
                .to_csv(OUT_DIR/f"{split}_WSI_risks_fold{fold_id}.csv",    index=False)
            logging.info(f"[✓] [{split}] CSV 已保存")
        # end split
    # end fold

# --------- 运行 ---------
run_LOPO()