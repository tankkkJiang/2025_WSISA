#!/usr/bin/env python
# -*- coding: utf-8 -*-
# WSISA/DeepConvSurv_pytorch.py
"""
* 去掉 theano / lasagne 依赖
* 通过参数传入数据、索引，不再硬编码路径
* 加入尽量详细的打印与异常捕获
"""

import os, time, math
import numpy as np
import pandas as pd
from PIL import Image
from typing import Sequence, List

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from lifelines.utils import concordance_index

# ----------------------------------------------------------
# 网络结构 & 损失（保持不变）
# ----------------------------------------------------------
class DeepSurv(nn.Module):
    def __init__(self, in_ch: int = 3):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, 32, 7, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 2),   nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2),   nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1)
        )
        self.head = nn.Linear(32, 1)

    def forward(self, x):                       # (N,C,H,W) → (N,1)
        return self.head(self.body(x).flatten(1))


class CoxPHLoss(nn.Module):
    """负对数偏似然 (Cox)"""
    def forward(self, risk, t, e):
        order = torch.argsort(t, descending=True)
        risk, t, e = risk[order], t[order], e[order]
        hr = risk.exp()
        log_cumsum = torch.log(torch.cumsum(hr, 0))
        return -torch.sum((risk - log_cumsum) * e) / (e.sum() + 1e-8)
# ----------------------------------------------------------


class DeepConvSurv:
    """封装训练 / 评估逻辑，便于外部脚本调用"""

    def __init__(self,
                 learning_rate: float,
                 channel: int,
                 width: int,
                 height: int,
                 lr_decay: float = 0.01):
        self.lr = learning_rate
        self.lr_decay = lr_decay

        # NLST 预估均值 / 方差，若需要可替换
        mean = [0.696, 0.590, 0.669]
        std  = [0.256, 0.294, 0.246]
        self.tr = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[INIT] DeepConvSurv on {self.device}")

    # ------------------------------------------------------
    # 公开接口：训练一个簇，返回验证 & 测试 C‑index
    # ------------------------------------------------------
    def fit(self,
            df_patch: pd.DataFrame,
            train_idx: Sequence[int],
            val_idx:   Sequence[int],
            test_idx:  Sequence[int],
            epochs: int = 20,
            batch_size: int = 30,
            verbose: bool = True) -> (float, float):

        if len(val_idx) == 0 or len(test_idx) == 0:
            raise ValueError("val_idx / test_idx 不能为空！")

        # ========= 数据准备 =========
        imgs = df_patch['patch_path'].tolist()
        t    = df_patch['surv'].astype(np.float32).values
        e    = df_patch['status'].astype(np.float32).values

        def make_loader(indices, shuffle):
            ds = torch.utils.data.TensorDataset(
                torch.arange(len(indices))      # 仅存索引，真正加载延迟到 __getitem__
            )
            return torch.utils.data.DataLoader(ds, batch_size=batch_size,
                                               shuffle=shuffle, num_workers=0,
                                               collate_fn=lambda x: x)

        loader_tr = make_loader(train_idx, shuffle=True)
        loader_va = make_loader(val_idx,   shuffle=False)
        loader_te = make_loader(test_idx,  shuffle=False)

        # ========= 网络 / 优化 =========
        sample_img = Image.open(os.path.join(os.getcwd(), imgs[0]))
        c = len(sample_img.getbands())      # 动态判断通道
        net = DeepSurv(c).to(self.device)
        optim_ = optim.Adam(net.parameters(), lr=self.lr)
        lossfn = CoxPHLoss()

        best_val_c, best_state = 0., None
        start = time.time()

        # ========= 训练循环 =========
        for ep in range(1, epochs + 1):
            net.train()
            ep_loss = 0.
            for batch in loader_tr:
                idx = [i.item() for i in batch]           # 索引列表
                xb, tb, eb = self._load_batch(idx, imgs, t, e)
                risk = net(xb)
                loss = lossfn(risk, tb, eb)

                optim_.zero_grad()
                loss.backward()
                optim_.step()

                ep_loss += loss.item()

            # 验证
            val_c = self._eval_cindex(net, loader_va, imgs, t, e)
            if val_c > best_val_c:
                best_val_c = val_c
                best_state = net.state_dict()

            if verbose:
                print(f"[Epoch {ep:02d}] loss={ep_loss/len(loader_tr):.4f} "
                      f"valC={val_c:.4f}")

            # 简单学习率衰减
            for g in optim_.param_groups:
                g['lr'] = self.lr / (1 + ep * self.lr_decay)

        print(f"[TRAIN] Done in {time.time()-start:.1f}s; "
              f"best val C‑index={best_val_c:.4f}")

        # ========= 测试 =========
        net.load_state_dict(best_state)
        test_c = self._eval_cindex(net, loader_te, imgs, t, e)
        print(f"[TEST]  final test C‑index = {test_c:.4f}")

        return best_val_c, test_c

    # ------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------
    def _load_batch(self,
                    idx_list: List[int],
                    img_paths: List[str],
                    t_arr: np.ndarray,
                    e_arr: np.ndarray):
        """一次性把一个 batch 的 多张图片 & 标签 读进 GPU"""
        xs, ts, es = [], [], []
        for i in idx_list:
            try:
                im = Image.open(os.path.join(os.getcwd(), img_paths[i])).convert('RGB')
                xs.append(self.tr(im))
                ts.append(t_arr[i])
                es.append(e_arr[i])
            except Exception as ex:
                print(f"[WARN] 读取失败 {img_paths[i]} → 跳过 ({ex})")
        xb = torch.stack(xs).to(self.device)
        tb = torch.tensor(ts, dtype=torch.float32, device=self.device)
        eb = torch.tensor(es, dtype=torch.float32, device=self.device)
        return xb, tb, eb

    def _eval_cindex(self, net, loader, img_paths, t_arr, e_arr) -> float:
        net.eval()
        all_risk, all_t, all_e = [], [], []
        with torch.no_grad():
            for batch in loader:
                idx = [i.item() for i in batch]
                xb, tb, eb = self._load_batch(idx, img_paths, t_arr, e_arr)
                all_risk.append(net(xb).cpu())
                all_t.append(tb.cpu()); all_e.append(eb.cpu())
        risk = torch.cat(all_risk).squeeze().numpy()
        tval = torch.cat(all_t).numpy()
        eval = torch.cat(all_e).numpy()
        try:
            return concordance_index(tval, -risk, eval)
        except ZeroDivisionError:
            return 0.5        # 无可比较对
# ----------------------------------------------------------