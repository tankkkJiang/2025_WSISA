#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_WSISA_selectedCluster_gpu_accum.py
  • 单折聚合（GPU 友好 · CPU 内存极限省）
  • sum_feat / sum_risk 全部在 GPU 上累加
  • 一次只加载一个 model 到 GPU；推完即释放
"""
import warnings, logging, time
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
    handlers=[logging.FileHandler("wsisa_gpu_accum.log"),
              logging.StreamHandler()]
)

# -------- 全局配置 --------
ROOT       = Path(__file__).resolve().parent
PATCH_CSV  = ROOT / "cluster_result" / "patches_1000_cls10_expanded.csv"
MODEL_DIR  = ROOT / "log" / "wsisa_patch10" / "convimgmodel"
SEL_FILE   = ROOT / "log" / "selected_clusters.txt"
OUT_DIR    = ROOT / "results"; OUT_DIR.mkdir(exist_ok=True, parents=True)

MEAN, STD   = [0.6964,0.5905,0.6692], [0.2559,0.2943,0.2462]
GPU_BATCH   = 8           # 进一步缩小，显存小可设置更小
CPU_WORKERS = 0           # 主进程加载，最省 CPU RAM
PIN_MEMORY  = False
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.info(f"DEVICE={DEVICE}, GPU_BATCH={GPU_BATCH}")

# -------- 读取数据 --------
selected_cluster = [int(x) for x in SEL_FILE.read_text().split(",") if x.strip().isdigit()]
logging.info(f"选簇: {selected_cluster}")

patch_df   = pd.read_csv(PATCH_CSV, usecols=["pid","surv","status","patch_path","cluster"])
patient_df = patch_df[["pid","surv","status"]].drop_duplicates()
logging.info(f"患者数={len(patient_df)}, Patch 总数={len(patch_df)}")

# -------- Dataset --------
TRANSF = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
class ClusterDataset(Dataset):
    def __init__(self, paths, idxs):
        self.paths = paths
        self.idxs  = idxs
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, i):
        p = ROOT / self.paths[i]
        img = Image.open(p).convert("RGB")
        t = TRANSF(img)
        img.close()
        return t, self.idxs[i]

# -------- 主流程 --------
def run_fold(fold_id, train_p, valid_p, test_p):
    for split, pids in dict(train=train_p, valid=valid_p, test=test_p).items():
        logging.info(f"=== [{split}] {len(pids)} 位患者 ===")
        df = patch_df[patch_df["pid"].isin(pids)].reset_index(drop=True)
        total_cnt = Counter(df["pid"])

        # 患者映射
        pid_list = pids
        pid2i = {pid:i for i,pid in enumerate(pid_list)}
        nC = len(selected_cluster)
        nP = len(pid_list)

        sum_feat = None     # later: torch.zeros(nC,nP,feat_dim,device=DEVICE)
        sum_risk = None     # later: torch.zeros(nC,nP,device=DEVICE)
        feat_dim = None

        t0 = time.time()
        for c_idx, c in enumerate(tqdm(selected_cluster, desc=f"[{split}] clusters")):
            idxs = np.where(df["cluster"].values == c)[0]
            if not len(idxs):
                continue

            # loader for this cluster
            paths = df.loc[idxs, "patch_path"].tolist()
            ds    = ClusterDataset(paths, idxs.tolist())
            loader= DataLoader(ds,
                               batch_size=GPU_BATCH,
                               shuffle=False,
                               num_workers=CPU_WORKERS,
                               pin_memory=PIN_MEMORY)

            # load model
            mpath = MODEL_DIR / f"convimgmodel_cluster{c}_fold{fold_id}.pth"
            if not mpath.exists():
                mpath = MODEL_DIR / f"convimgmodel_cluster{c}_fold1.pth"
            model = DeepConvSurv(get_features=True)
            model.load_state_dict(torch.load(mpath, map_location="cpu")["model"])
            model.to(DEVICE).eval()

            # batch 推理 & GPU 累加
            for imgs, glb_idxs in loader:
                imgs = imgs.to(DEVICE, non_blocking=True)
                fts, rks = model(imgs)            # fts: (B,feat_dim), rks: (B,)
                if sum_feat is None:
                    feat_dim = fts.shape[1]
                    sum_feat = torch.zeros(nC, nP, feat_dim, device=DEVICE)
                    sum_risk = torch.zeros(nC, nP, device=DEVICE)

                # 把 glb_idxs 映射到 pid idx
                pidxs = [ pid2i[df.at[int(g),"pid"]] for g in glb_idxs ]
                pidxs = torch.tensor(pidxs, device=DEVICE)

                # 对每个 pid 分组累加
                for pid_idx in torch.unique(pidxs):
                    mask = pidxs == pid_idx
                    sum_feat[c_idx, pid_idx] += fts[mask].sum(dim=0)
                    sum_risk[c_idx, pid_idx] += rks[mask].sum()

            # 卸载模型
            model.cpu(); del model; torch.cuda.empty_cache()
            logging.info(f"簇 {c} 完成, patch={len(idxs)}")

        logging.info(f"[{split}] 推理结束, 用时 {time.time()-t0:.1f}s")

        # 从 GPU 拉回 CPU，做加权均值并存 CSV
        sf = sum_feat.cpu().numpy()   # (C,P,D)
        sr = sum_risk.cpu().numpy()   # (C,P)

        rows = []
        for pid_idx, pid in enumerate(pid_list):
            surv   = int(df[df["pid"]==pid]["surv"].iat[0])
            status = int(df[df["pid"]==pid]["status"].iat[0])
            tot    = total_cnt[pid]
            fea = []
            rsk = []
            for c_idx in range(nC):
                block = sf[c_idx,pid_idx]
                fea.extend((block / tot).tolist())
                rsk.append((sr[c_idx,pid_idx] / tot).item())
            rows.append({
                "pid":pid, "surv":surv, "status":status,
                **{f"fea_{i}":v for i,v in enumerate(fea)},
                **{f"risk_{j}":v for j,v in enumerate(rsk)}
            })

        out = pd.DataFrame(rows)
        # 保存
        feat_cols = [f"fea_{i}"  for i in range(nC*feat_dim)]
        risk_cols = [f"risk_{j}" for j in range(nC)]
        out[["pid","surv","status"]+feat_cols]\
            .to_csv(OUT_DIR/f"{split}_patient_features_fold{fold_id}.csv", index=False)
        out[["pid","surv","status"]+risk_cols]\
            .to_csv(OUT_DIR/f"{split}_patient_risks_fold{fold_id}.csv",    index=False)
        logging.info(f"[✓] [{split}] CSV 保存完毕")

# ---------- 执行 ---------
pids = patient_df["pid"].tolist()
run_fold(1, train_p=pids[2:], valid_p=[pids[1]], test_p=[pids[0]])