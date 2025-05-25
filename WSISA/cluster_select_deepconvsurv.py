#!/usr/bin/env python
"""
WSISA/clyster_select_deepconvsurv.py

Model‑selection script for DeepConvSurv under the **new WSISA repo layout**.
The training / validation logic is **unchanged** – only paths, default names
and a few hard‑coded placeholders were updated so that the script runs
out‑of‑the‑box with the following structure:

WSISA/
├── data/patches/…
├── data/patients.csv
├── cluster_result/patches_1000_cls10_expanded.csv
└── …
"""
import os
import numpy as np
import pandas as pd
from PIL import Image

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

import DeepConvSurv_pytorch as deep_conv_surv  # still lives in repo root

# ==================== Hyper‑parameters ====================
MODEL          = "deepconvsurv"       # keep original logic/choices
EPOCHS         = 20
LR             = 5e-4
SEED           = 1
BATCH_SIZE     = 30

# ==================== Repository paths ====================
BASE_DIR            = os.path.abspath(os.path.dirname(__file__))
PATCHES_ROOT        = os.path.join(BASE_DIR, "data", "patches")
PATIENT_LABEL_CSV   = os.path.join(BASE_DIR, "data", "patients.csv")
EXP_LABEL_CSV       = os.path.join(
    BASE_DIR, "cluster_result", "patches_1000_cls10_expanded.csv"
)
LOG_DIR             = os.path.join(BASE_DIR, "log", "wsisa_patch10")
os.makedirs(LOG_DIR, exist_ok=True)

# ==================== Helpers ====================

def convert_index(pid_list: np.ndarray, expand_df: pd.DataFrame):
    """Map a list of patient IDs to *patch‑level* row indices in `expand_df`."""
    idx_per_patient = [expand_df.index[expand_df["pid"] == pid].tolist()
                       for pid in pid_list]
    counts = [len(i) for i in idx_per_patient]
    flat   = [i for sub in idx_per_patient for i in sub]
    return flat, counts

# ==========================================================

def model_selection(
    patches_root: str = PATCHES_ROOT,
    label_path: str = PATIENT_LABEL_CSV,
    expand_label_path: str = EXP_LABEL_CSV,
    train_test_ratio: float = 0.9,
    train_valid_ratio: float = 0.9,
    *,
    seed: int = SEED,
    model: str = MODEL,
    batchsize: int = BATCH_SIZE,
    epochs: int = EPOCHS,
    lr: float = LR,
    **kwargs,
):
    """Wrap original pipeline with new‑path defaults; logic untouched."""
    print("\n--------------------- Model Selection ---------------------")
    print(f"Training Model: {model}")
    print("epochs:", epochs, " tr/test ratio:", train_test_ratio,
          " tr/val ratio:", train_valid_ratio)
    print("learning rate:", lr, " batch size:", batchsize)
    print("-----------------------------------------------------------\n")

    # ---------- Load label tables ----------
    labels_df       = pd.read_csv(label_path)
    expand_label_df = pd.read_csv(expand_label_path)

    # derive cluster id from path (e.g. …cls10_expanded.csv → 10)
    try:
        cluster_id = int(os.path.basename(expand_label_path).split("cls")[-1].split(".")[0])
    except Exception:
        cluster_id = -1  # fallback; does not affect original logic

    e_status = labels_df["status"].values  # for stratification
    skf      = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    test_ci_scores = []
    fold_idx = 1
    for train_pid_idx, test_pid_idx in skf.split(np.zeros(len(e_status)), e_status):
        print(f"\n================ Fold {fold_idx} =================")

        # ----- split patient IDs -----
        train_pids = labels_df["pid"].iloc[train_pid_idx].values
        test_pids  = labels_df["pid"].iloc[test_pid_idx].values

        # ----- inner split for validation -----
        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=1 - train_valid_ratio,
            random_state=seed,
        )
        inner_train_idx, inner_val_idx = next(sss.split(np.zeros(len(train_pid_idx)), e_status[train_pid_idx]))
        inner_train_pids = train_pids[inner_train_idx]
        inner_val_pids   = train_pids[inner_val_idx]

        # map to patch‑level indices
        train_idx, _ = convert_index(inner_train_pids, expand_label_df)
        valid_idx, _ = convert_index(inner_val_pids,   expand_label_df)
        test_idx,  _ = convert_index(test_pids,        expand_label_df)

        # save indices (same names/pattern as old version)
        np.savetxt(os.path.join(LOG_DIR, f"train_cluster{cluster_id}_fold{fold_idx}.csv"),
                   train_idx, delimiter=",", header="index", comments="")
        np.savetxt(os.path.join(LOG_DIR, f"valid_cluster{cluster_id}_fold{fold_idx}.csv"),
                   valid_idx, delimiter=",", header="index", comments="")
        np.savetxt(os.path.join(LOG_DIR, f"test_cluster{cluster_id}_fold{fold_idx}.csv"),
                   test_idx,  delimiter=",", header="index", comments="")

        # ----- infer input image shape from *first* patch -----
        sample_patch_rel = expand_label_df["patch_path"].iloc[0]
        sample_patch_abs = os.path.join(BASE_DIR, sample_patch_rel)
        img = Image.open(sample_patch_abs)
        width, height = img.size
        channel       = len(img.getbands())

        if model == "deepconvsurv":
            net = deep_conv_surv.DeepConvSurv(
                learning_rate=lr,
                channel=channel,
                width=width,
                height=height,
            )
            # ORIGINAL ARG ORDER PRESERVED (test/valid swapped intentionally)
            ci = net.train(
                data_path=patches_root,
                label_path=expand_label_path,
                train_index=train_idx,
                test_index=valid_idx,
                valid_index=test_idx,
                model_index=fold_idx,
                cluster=cluster_id,
                batch_size=batchsize,
                ratio=train_test_ratio,
                num_epochs=epochs,
            )
            test_ci_scores.append(ci)
        else:
            print("[Warn] unsupported model; skipping fold.")

        fold_idx += 1

    print("\n================ Overall =================")
    print(f"C‑index mean: {np.mean(test_ci_scores):.4f}  std: {np.std(test_ci_scores):.4f}")


# ==================== CLI entry ====================
if __name__ == "__main__":
    model_selection()
else:
    print("cluster_select_deepconvsurv_pytorch imported")