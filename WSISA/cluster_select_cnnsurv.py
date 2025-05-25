#!/usr/bin/env python
# cluster_select_cnnsurv.py
import os, time, random
import numpy as np, pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

import torch, torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

from cnn_survival import CNNSurv, CoxPHLoss, c_index_torch


# ----------------------------------------------------------
BASE   = os.path.abspath(os.path.dirname(__file__))
PATCHES_ROOT      = os.path.join(BASE, "data", "patches")
PATIENT_LABEL_CSV = os.path.join(BASE, "data", "patients.csv")
EXP_LABEL_CSV     = os.path.join(BASE, "cluster_result",
                                 "patches_1000_cls10_expanded.csv")
LOG_DIR = os.path.join(BASE, "log", "wsisa_patch10_pt")
os.makedirs(LOG_DIR, exist_ok=True)

EPOCHS, BATCH, LR, SEED = 20, 30, 5e-4, 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
# ----------------------------------------------------------

class PatchDataset(Dataset):
    def __init__(self, df_idx, root, label_df, mean=None, std=None):
        self.df = df_idx
        self.root = root
        self.label_df = label_df
        self.tr = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean or [0.5]*3, std=std or [0.5]*3)
        ])

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(BASE, row.patch_path)   # patch_path 列相对仓库根
        img = Image.open(img_path).convert('RGB')
        x = self.tr(img)
        return (x,
                torch.tensor(row.surv,   dtype=torch.float32),
                torch.tensor(row.status, dtype=torch.float32))


def convert_index(pids, full_df):
    idx = full_df.index[full_df.pid.isin(pids)].tolist()
    return idx


def run_fold(train_pids, val_pids, test_pids,
             expand_df, fold_id):

    train_idx = convert_index(train_pids, expand_df)
    val_idx   = convert_index(val_pids,   expand_df)
    test_idx  = convert_index(test_pids,  expand_df)

    # 保存索引方便复现
    np.savetxt(os.path.join(LOG_DIR, f"train_fold{fold_id}.csv"),
               train_idx, delimiter=',', header='index', comments='')
    np.savetxt(os.path.join(LOG_DIR, f"valid_fold{fold_id}.csv"),
               val_idx, delimiter=',', header='index', comments='')
    np.savetxt(os.path.join(LOG_DIR, f"test_fold{fold_id}.csv"),
               test_idx,  delimiter=',', header='index', comments='')

    # Dataset / Loader
    train_ds = PatchDataset(expand_df.loc[train_idx], PATCHES_ROOT, expand_df)
    val_ds   = PatchDataset(expand_df.loc[val_idx],   PATCHES_ROOT, expand_df)
    test_ds  = PatchDataset(expand_df.loc[test_idx],  PATCHES_ROOT, expand_df)

    train_ld = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=4)
    val_ld   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=4)
    test_ld  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False, num_workers=4)

    # 网络 & 优化器
    c, w, h = next(iter(train_ld))[0].shape[1:]
    model = CNNSurv(c).to(DEVICE)
    criterion = CoxPHLoss()
    optim = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_c = 0
    for ep in range(1, EPOCHS+1):
        model.train()
        for x, t, e in train_ld:
            x, t, e = x.to(DEVICE), t.to(DEVICE), e.to(DEVICE)
            loss = criterion(model(x), t, e)
            optim.zero_grad(); loss.backward(); optim.step()

        # 验证
        model.eval(); risk_all = []; t_all = []; e_all = []
        with torch.no_grad():
            for x, t, e in val_ld:
                x, t, e = x.to(DEVICE), t.to(DEVICE), e.to(DEVICE)
                risk_all.append(model(x)); t_all.append(t); e_all.append(e)
        risk_all = torch.cat(risk_all); t_all = torch.cat(t_all); e_all = torch.cat(e_all)
        val_c = c_index_torch(risk_all, t_all, e_all)
        best_val_c = max(best_val_c, val_c)
        print(f"Fold{fold_id}  Epoch{ep:02d}  val C‑idx={val_c:.4f}")

    # --- 测试 ---
    model.eval(); risk_all = []; t_all = []; e_all = []
    with torch.no_grad():
        for x, t, e in test_ld:
            x, t, e = x.to(DEVICE), t.to(DEVICE), e.to(DEVICE)
            risk_all.append(model(x)); t_all.append(t); e_all.append(e)
    risk_all = torch.cat(risk_all); t_all = torch.cat(t_all); e_all = torch.cat(e_all)
    test_c = c_index_torch(risk_all, t_all, e_all)
    print(f"Fold{fold_id}  **TEST C‑index = {test_c:.4f}**\n")
    return test_c


def main():
    expand_df = pd.read_csv(EXP_LABEL_CSV)   # 必含 pid / surv / status / patch_path
    labels_df = expand_df[['pid', 'surv', 'status']].drop_duplicates().reset_index(drop=True)

    y_status = labels_df.status.values
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)

    test_scores = []
    for fold, (tr_pid_idx, te_pid_idx) in enumerate(skf.split(np.zeros(len(y_status)), y_status), 1):
        train_pids = labels_df.pid.iloc[tr_pid_idx].values
        test_pids  = labels_df.pid.iloc[te_pid_idx].values

        # 再划验证集
        sss = StratifiedShuffleSplit(1, test_size=0.1, random_state=SEED)
        tr_inner, val_inner = next(sss.split(np.zeros(len(tr_pid_idx)), y_status[tr_pid_idx]))
        val_pids = train_pids[val_inner]
        train_pids = train_pids[tr_inner]

        score = run_fold(train_pids, val_pids, test_pids, expand_df, fold)
        test_scores.append(score)

    print("========== Overall ==========")
    print(f"C‑index mean={np.mean(test_scores):.4f}  std={np.std(test_scores):.4f}")


if __name__ == "__main__":
    main()