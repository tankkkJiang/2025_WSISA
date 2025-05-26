#!/usr/bin/env python
# -*- coding: utf‑8 -*-
"""
WSISA/networks.py

纯 PyTorch 版 DeepConvSurv 及其损失, 兼容“提特征 + 出风险分数”两种模式
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from lifelines.utils import concordance_index


def c_index_torch(risk_pred, t, e):
    """
    将 torch.Tensor 风险分数 / 生存时间 / 状态 转成 numpy，然后调用 lifelines 计算 C-index
    """
    # 把 (N,1) → (N,)
    risk = risk_pred.detach().cpu().view(-1).numpy()
    t    = t.detach().cpu().numpy()
    e    = e.detach().cpu().numpy()
    # lifelines 的 concordance_index 接受 (预测值, 时间, 事件)
    return concordance_index(-risk, t, e)


# ----------- 网络本体 -----------
class DeepConvSurv(nn.Module):
    """
    - 若构造时 get_features=True: forward 返回 (feat_vec, risk)
    - 否则仅返回 risk
    """
    def __init__(self, in_channels: int = 3,
                 get_features: bool = False) -> None:
        super().__init__()
        self.get_features = get_features

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=7, stride=3), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),      # → (N,32,1,1)
            nn.Flatten()                  # → (N,32)
        )
        self.fc = nn.Linear(32, 1)        # risk (= log‑hazard)

    # ----------- 前向 -----------
    def forward(self, x):
        feat = self.features(x)
        risk = self.fc(feat)
        if self.get_features:
            return feat, risk
        return risk


# ----------- Cox 部分似然负对数损失 -----------
class NegativeLogLikelihood(nn.Module):
    """
    输入
    ----
    risk_pred : (N,1) tensor, 越大→风险越高 (log‑hazard)
    t         : (N,)  存活时间
    e         : (N,)  结局; 1=死亡 / 0=删失
    """
    def forward(self, risk_pred, t, e):
        # 按 t 降序
        order = torch.argsort(t, descending=True)
        eta   = risk_pred.view(-1)[order]
        e     = e[order]
        # 累积求和的 log(exp) = logcumexp
        log_cum_hazard = torch.logcumsumexp(eta, dim=0)
        loss = -torch.sum((eta - log_cum_hazard) * e) / (e.sum() + 1e-8)
        return loss


# ----------- 计算 C‑index（torch→numpy）-----------
def c_index_torch(risk, t, e):
    risk = risk.detach().cpu().view(-1).numpy()
    t    = t.detach().cpu().numpy()
    e    = e.detach().cpu().numpy()
    return concordance_index(-risk, t, e)