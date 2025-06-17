import torch
import torch.nn as nn
import torch.nn.functional as F

class FrequencyDomainL1Loss(nn.Module):
    """
    同时计算时域 L1 与加权频域 L1 的组合损失

    参数
    -------
    alpha : float, 默认 0.5
        时域损失占比；(1‑alpha) 用于频域分量
    freq_mode : {'log','gaussian','parabolic','linear','uniform'}
        生成频率权重的方式（见 _get_frequency_weights）
    reduction : {'mean','sum','none'}
        与 nn.*Loss 一致
    """
    def __init__(self, alpha: float = 0.5,
                 freq_mode: str = 'log',
                 reduction: str = 'mean'):
        super().__init__()
        assert 0.0 <= alpha <= 1.0, "`alpha` 必须在 [0,1]"
        assert reduction in ('mean', 'sum', 'none')
        self.alpha = alpha
        self.freq_mode = freq_mode
        self.reduction = reduction

    # ------------------------------------------------------------
    # forward
    # ------------------------------------------------------------
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        y_pred / y_true : (B, T, C)  或  (B, T) 皆可
        """
        if y_pred.shape != y_true.shape:
            raise ValueError(f"预测与标签形状不一致: {y_pred.shape} vs {y_true.shape}")

        # ---- 时域 L1 ----
        loss_time = (y_pred - y_true).abs()

        # ---- 频域 L1 (加权) ----
        y_pred_fft = torch.fft.rfft(y_pred, dim=1)
        y_true_fft = torch.fft.rfft(y_true, dim=1)
        freq_diff = (y_pred_fft - y_true_fft).abs()        # (B, F, C)

        F_bins = freq_diff.shape[1]
        weights = self._get_frequency_weights(F_bins,
                                              device=y_pred.device,
                                              mode=self.freq_mode)   # (1, F, 1)
        loss_freq = freq_diff * weights

        # ---- 汇总 ----
        if self.reduction == 'mean':
            loss_time = loss_time.mean()
            loss_freq = loss_freq.mean()
        elif self.reduction == 'sum':
            loss_time = loss_time.sum()
            loss_freq = loss_freq.sum()
            # 'none' 不变形

        loss = self.alpha * loss_time + (1 - self.alpha) * loss_freq
        return loss

    # ------------------------------------------------------------
    # utils
    # ------------------------------------------------------------
    @staticmethod
    def _get_frequency_weights(F: int,
                               device: torch.device,
                               mode: str = 'log') -> torch.Tensor:
        """
        生成形状 (1, F, 1) 的权重张量
        """
        if mode == 'gaussian':
            x = torch.linspace(0, 1, F, device=device)
            mu, sigma = 0.5, 0.15
            w = torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

        elif mode == 'parabolic':
            x = torch.linspace(0, 1, F, device=device)
            w = x ** 2

        elif mode == 'log':
            x = torch.linspace(1, F, F, device=device)
            w = torch.log(x)
            w /= w.max()

        elif mode == 'linear':
            w = torch.linspace(1.0, 2.0, F, device=device)

        else:   # 'uniform' or fallback
            w = torch.ones(F, device=device)

        w[0] = 0.0                             # 屏蔽直流分量 (0 Hz)
        return w.view(1, F, 1)
