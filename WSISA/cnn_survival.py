# WSISA/cnn_survival.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from lifelines.utils import concordance_index
import numpy as np


class CoxPHLoss(nn.Module):
    """Negative log‑partial‑likelihood of Cox proportional‑hazards model."""
    def forward(self, risk, t, e):
        # 按生存时间降序排序
        order = torch.argsort(t, descending=True)
        risk, t, e = risk[order], t[order], e[order]
        # 累加 log‑cumsum‑exp trick 保数值稳定
        hazard_ratio = risk.exp()
        log_cumsum = torch.log(torch.cumsum(hazard_ratio, dim=0))
        loss = -torch.sum((risk - log_cumsum) * e) / e.sum()
        return loss


class CNNSurv(nn.Module):
    """简单 3‑层 CNN，输出 1 个 risk 分数"""
    def __init__(self, in_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 7, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 2),  nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2),  nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(32, 1)

    def forward(self, x):
        x = self.net(x).flatten(1)
        return self.head(x)


@torch.no_grad()
def c_index_torch(risk, t, e):
    """risk, t, e 均为 tensor，返回 concordance‑index(float)"""
    return concordance_index(
        t.cpu().numpy(),           # 时间
        (-risk).cpu().numpy(),     # 越大风险越大 ⇒ 取负号
        e.cpu().numpy(),           # 终点(1=事件)
    )