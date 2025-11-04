import math
from typing import Literal, Optional

import torch
from torch import nn


def _directional_cumsum(tensor: torch.Tensor, dim: int, reverse: bool = False) -> torch.Tensor:
    if reverse:
        tensor = torch.flip(tensor, dims=[dim])
    tensor = torch.cumsum(tensor, dim=dim)
    if reverse:
        tensor = torch.flip(tensor, dims=[dim])
    return tensor


class SelectiveScan2D(nn.Module):
    """Simplified SS2D-style selective scan for 2D feature maps."""

    def __init__(
        self,
        channels: int,
        hidden_ratio: float = 1.0,
        dropout: float = 0.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        hidden_channels = max(8, int(math.ceil(channels * hidden_ratio)))
        self.proj_in = nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False)
        self.gate = nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=True)

        self.scale = nn.Parameter(torch.ones(4, hidden_channels))
        self.norm = nn.GroupNorm(num_groups=1, num_channels=hidden_channels, eps=eps)
        self.activation = nn.SiLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.proj_out = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_proj = self.proj_in(x)
        gate = torch.sigmoid(self.gate(x))
        gated = x_proj * gate

        # four directional cumsums (left->right, right->left, top->bottom, bottom->top)
        lr = _directional_cumsum(gated, dim=-1, reverse=False)
        rl = _directional_cumsum(gated, dim=-1, reverse=True)
        tb = _directional_cumsum(gated, dim=-2, reverse=False)
        bt = _directional_cumsum(gated, dim=-2, reverse=True)

        stack = torch.stack([lr, rl, tb, bt], dim=0)
        fused = (stack * self.scale.view(4, 1, -1, 1, 1)).sum(dim=0) / 4.0

        fused = self.norm(fused)
        fused = self.activation(fused)
        fused = self.dropout(fused)
        fused = self.proj_out(fused)
        return fused + residual


class ChannelSE3D(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        reduction = max(1, reduction)
        reduced = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Conv3d(channels, reduced, kernel_size=1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv3d(reduced, channels, kernel_size=1, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.avg_pool(x)
        attn = self.fc1(attn)
        attn = self.act(attn)
        attn = self.fc2(attn)
        attn = self.gate(attn)
        return x * attn


class TriPlaneVMambaBlock(nn.Module):
    """Tri-plane SS2D block providing global context for 3D tensors."""

    def __init__(
        self,
        channels: int,
        hidden_ratio: float = 0.5,
        dropout: float = 0.0,
        fuse_mode: Literal["sum", "concat"] = "concat",
        use_se: bool = True,
        se_reduction: int = 8,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.fuse_mode = fuse_mode

        self.axial_scan = SelectiveScan2D(channels, hidden_ratio, dropout)
        self.coronal_scan = SelectiveScan2D(channels, hidden_ratio, dropout)
        self.sagittal_scan = SelectiveScan2D(channels, hidden_ratio, dropout)

        fuse_channels = channels if fuse_mode == "sum" else channels * 3
        self.fuse = nn.Sequential(
            nn.Conv3d(fuse_channels, channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(channels, affine=True),
            nn.SiLU(inplace=True),
        )

        self.se = ChannelSE3D(channels, reduction=se_reduction) if use_se else nn.Identity()
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

    def _apply_axial(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        axial = x.permute(0, 2, 1, 3, 4).contiguous().view(b * d, c, h, w)
        axial = self.axial_scan(axial)
        axial = axial.view(b, d, c, h, w).permute(0, 2, 1, 3, 4).contiguous()
        return axial

    def _apply_coronal(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        coronal = x.permute(0, 3, 1, 2, 4).contiguous().view(b * h, c, d, w)
        coronal = self.coronal_scan(coronal)
        coronal = coronal.view(b, h, c, d, w).permute(0, 2, 3, 1, 4).contiguous()
        return coronal

    def _apply_sagittal(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        sagittal = x.permute(0, 4, 1, 2, 3).contiguous().view(b * w, c, d, h)
        sagittal = self.sagittal_scan(sagittal)
        sagittal = sagittal.view(b, w, c, d, h).permute(0, 2, 3, 4, 1).contiguous()
        return sagittal

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        axial = self._apply_axial(x)
        coronal = self._apply_coronal(x)
        sagittal = self._apply_sagittal(x)

        if self.fuse_mode == "sum":
            fused = (axial + coronal + sagittal) / 3.0
        else:
            fused = torch.cat([axial, coronal, sagittal], dim=1)

        fused = self.fuse(fused)
        fused = self.se(fused)
        fused = self.dropout(fused)
        return fused


__all__ = ["SelectiveScan2D", "TriPlaneVMambaBlock", "ChannelSE3D"]

