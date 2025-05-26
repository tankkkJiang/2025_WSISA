#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WSISA/cluster_select_deepconvsurv.py   (简化版，仅 7 位病人一次性跑通)

* 读取 cluster_result/patches_1000_cls10_expanded.csv
* 按患者 status 分层拆成 3(train)+2(valid)+2(test) 位病人
* 对每个簇分别训练 DeepConvSurv，打印并保存模型
* 计算 valid/test C-index；遇到 "No admissable pairs" 捕获并置 0
* 输出选簇列表(log/selected_clusters.txt) — C-index ≥ 0.5
"""
import random, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch, torch.optim as optim
import torchvision.transforms as T
from tqdm import tqdm
from lifelines.utils import concordance_index

from networks import DeepConvSurv, NegativeLogLikelihood

warnings.filterwarnings("ignore", category=UserWarning)

# ------------ 常量 ------------
SEED = 1
random.seed(SEED)
torch.manual_seed(SEED)

ROOT        = Path(__file__).resolve().parent
CSV_PATH    = ROOT / "cluster_result" / "patches_1000_cls10_expanded.csv"
MODEL_DIR   = ROOT / "log" / "wsisa_patch10" / "convimgmodel"
SEL_FILE    = ROOT / "log" / "selected_clusters.txt"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS, BATCH, LR = 3, 64, 1e-4          # 为了演示, epoch 减到 3
C_THRESH          = 0.50
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEAN = [0.6964, 0.5905, 0.6692]
STD  = [0.2559, 0.2943, 0.2462]
TRANS = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])

# ------------ 数据加载 ------------
df = pd.read_csv(CSV_PATH)
print(f"\n>>> 共有 patch {len(df)} 条, 患者 {df['pid'].nunique()} 位")
print(">>> cluster 分布:")
for c, g in df.groupby("cluster"):
    print(f"  cluster {c:2d}: {len(g)} patches")

# ------------ 病人分层拆分 ------------
patients = df[["pid", "status"]].drop_duplicates()
dead  = patients[patients.status == 1]["pid"].tolist()
alive = patients[patients.status == 0]["pid"].tolist()

random.shuffle(dead);  random.shuffle(alive)
# 简单策略: 先确保每组至少 1 dead, 其余随机补足
train_p = [dead.pop()] if dead else []
valid_p = [dead.pop()] if dead else []
test_p  = [dead.pop()] if dead else []

# 补充剩余名额 (3-2-2)
def fill(target, n):
    while len(target) < n and alive:
        target.append(alive.pop())
    while len(target) < n and dead:
        target.append(dead.pop())

fill(train_p, 3); fill(valid_p, 2); fill(test_p, 2)
# 若还没凑够 (数据非常偏)，就随便塞
remain = dead + alive
random.shuffle(remain)
fill(train_p, 3); fill(valid_p, 2); fill(test_p, 2)

print("\n>>> 病人划分:")
print("  train:", train_p)
print("  valid:", valid_p)
print("  test :", test_p)

def idx_of(pids):
    return df.index[df.pid.isin(pids)].tolist()

train_idx, valid_idx, test_idx = map(idx_of, (train_p, valid_p, test_p))
clusters = sorted(df.cluster.unique())
fold_cidx = {c: [] for c in clusters}

# ------------ utils ------------
def make_dataset(rows):
    xs, ts, es = [], [], []
    for _, r in rows.iterrows():
        xs.append(TRANS(Image.open(ROOT / r.patch_path)))
        ts.append(float(r.surv)); es.append(float(r.status))
    return (
        torch.stack(xs),
        torch.tensor(ts, dtype=torch.float32),
        torch.tensor(es, dtype=torch.float32),
    )

def ci_torch(risk, t, e):
    try:
        return concordance_index(-risk.detach().cpu().view(-1).numpy(),
                                 t.detach().cpu().numpy(),
                                 e.detach().cpu().numpy())
    except ZeroDivisionError:
        return 0.0

# ------------ 逐簇训练 ------------
crit = NegativeLogLikelihood()
for c in tqdm(clusters, desc="train clusters"):
    tr = df.loc[train_idx].query("cluster == @c")
    va = df.loc[valid_idx].query("cluster == @c")
    te = df.loc[test_idx].query("cluster == @c")

    if len(tr) < 5:             # demo, 样本太少直接跳
        print(f"[Skip] cluster {c} 训练样本 <5")
        fold_cidx[c].append(0.0)
        continue

    Xtr,Ttr,Etr = make_dataset(tr)
    Xva,Tva,Eva = make_dataset(va)
    Xte,Tte,Ete = make_dataset(te)

    model = DeepConvSurv(get_features=False).to(DEVICE)
    opt   = optim.Adam(model.parameters(), lr=LR)

    best_vc = -1
    for ep in range(1, EPOCHS+1):
        model.train()
        perm = torch.randperm(Xtr.size(0))
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i+BATCH]
            x = Xtr[idx].to(DEVICE); t=Ttr[idx].to(DEVICE); e=Etr[idx].to(DEVICE)
            loss = crit(model(x), t, e)
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval(); vc = ci_torch(model(Xva.to(DEVICE)), Tva, Eva)
        print(f"cluster {c}  epoch {ep}/{EPOCHS}  val‑C={vc:.3f}")
        if vc > best_vc:
            best_vc = vc
            torch.save({"model": model.state_dict()},
                       MODEL_DIR / f"convimgmodel_cluster{c}.pth")

    model.eval(); tc = ci_torch(model(Xte.to(DEVICE)), Tte, Ete)
    fold_cidx[c].append(tc)
    print(f"[Done] cluster {c}: best‑val C={best_vc:.3f}, test C={tc:.3f}")

# ------------ 选簇 ------------
selected = [c for c, v in fold_cidx.items() if np.mean(v) >= C_THRESH]
print("\n>>> 满足 C-index≥%.2f 的簇: %s" % (C_THRESH, selected))
SEL_FILE.write_text(",".join(map(str, selected)))
print(f"[Saved] 选簇列表 → {SEL_FILE}")