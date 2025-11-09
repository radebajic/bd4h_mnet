import math
from typing import Literal, Optional

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _scan_chunk(x_chunk, delta_chunk, B_chunk, C_chunk, A_expanded, h_init):
    """Process a small chunk - used with gradient checkpointing."""
    chunk_len = x_chunk.shape[1]
    h = h_init
    outputs = []
    
    for t in range(chunk_len):
        x_t = x_chunk[:, t]
        delta_t = delta_chunk[:, t]
        B_t = B_chunk[:, t]
        C_t = C_chunk[:, t]
        
        deltaA_t = torch.exp(delta_t.unsqueeze(-1) * A_expanded)
        deltaB_t = delta_t.unsqueeze(-1) * B_t.unsqueeze(1)
        h = deltaA_t * h + deltaB_t * x_t.unsqueeze(-1)
        
        y_t = (h * C_t.unsqueeze(1)).sum(dim=-1)
        outputs.append(y_t)
    
    return torch.stack(outputs, dim=1), h


def selective_scan_1d(x: torch.Tensor, delta: torch.Tensor, A: torch.Tensor, 
                      B_ssm: torch.Tensor, C_ssm: torch.Tensor, D_param: torch.Tensor,
                      chunk_size: int = 256) -> torch.Tensor:
    """
    Memory-efficient selective scan with gradient checkpointing.
    Processes in small chunks to limit autograd graph depth.
    
    Args:
        x: (B, L, D) input
        delta: (B, L, D) step size
        A: (D, N) state transition
        B_ssm: (B, L, N) input matrix
        C_ssm: (B, L, N) output matrix
        D_param: (D,) skip connection
        chunk_size: Process this many tokens per checkpoint (smaller = less memory)
        
    Returns:
        y: (B, L, D) output
    """
    B_batch, L, D_dim = x.shape
    N = A.shape[1]
    
    # Initialize state
    h = torch.zeros(B_batch, D_dim, N, device=x.device, dtype=x.dtype)
    A_expanded = A.unsqueeze(0)
    
    # Process in chunks with gradient checkpointing
    outputs = []
    num_chunks = (L + chunk_size - 1) // chunk_size
    
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, L)
        
        # Get chunk
        x_chunk = x[:, start_idx:end_idx]
        delta_chunk = delta[:, start_idx:end_idx]
        B_chunk = B_ssm[:, start_idx:end_idx]
        C_chunk = C_ssm[:, start_idx:end_idx]
        
        # Checkpoint this chunk (trades compute for memory)
        if x.requires_grad:
            y_chunk, h = checkpoint(_scan_chunk, x_chunk, delta_chunk, B_chunk, 
                                   C_chunk, A_expanded, h, use_reentrant=False)
        else:
            y_chunk, h = _scan_chunk(x_chunk, delta_chunk, B_chunk, C_chunk, A_expanded, h)
        
        outputs.append(y_chunk)
    
    # Concatenate chunks
    y = torch.cat(outputs, dim=1)
    
    # Skip connection
    y = y + x * D_param.unsqueeze(0).unsqueeze(0)
    
    return y


class SelectiveScan2D(nn.Module):
    """True selective scan with state-space modeling for 2D feature maps."""

    def __init__(
        self,
        channels: int,
        hidden_ratio: float = 1.0,
        dropout: float = 0.0,
        d_state: int = 16,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.d_state = d_state
        hidden_channels = max(8, int(math.ceil(channels * hidden_ratio)))
        
        # Input projection
        self.proj_in = nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False)
        
        # SSM parameters for 4 directions
        self.num_directions = 4
        
        # A: state transition matrix (log space for stability)
        self.A_log = nn.Parameter(torch.randn(self.num_directions, hidden_channels, d_state))
        
        # D: skip connection
        self.D = nn.Parameter(torch.ones(self.num_directions, hidden_channels))
        
        # Projects for computing B, C, delta (data-dependent)
        self.x_proj = nn.ModuleList([
            nn.Conv2d(hidden_channels, d_state + d_state + hidden_channels, kernel_size=1)
            for _ in range(self.num_directions)
        ])
        
        self.norm = nn.GroupNorm(num_groups=1, num_channels=hidden_channels, eps=eps)
        self.activation = nn.SiLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.proj_out = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)
        
        # Learnable direction fusion weights
        self.direction_weights = nn.Parameter(torch.ones(self.num_directions) / self.num_directions)

    def scan_direction(self, x: torch.Tensor, direction_idx: int) -> torch.Tensor:
        """Scan in one direction with SSM."""
        B, C_dim, H, W = x.shape
        
        # Flatten spatial dimensions for scanning
        x_flat = x.flatten(2).transpose(1, 2)  # (B, H*W, C_dim)
        
        # Get data-dependent parameters
        params = self.x_proj[direction_idx](x)  # (B, d_state*2 + C_dim, H, W)
        params_flat = params.flatten(2).transpose(1, 2)  # (B, H*W, d_state*2 + C_dim)
        
        B_ssm, C_ssm, delta = params_flat.split([self.d_state, self.d_state, C_dim], dim=-1)
        delta = F.softplus(delta)  # Ensure positive
        
        # Get SSM matrices (negative for stability)
        A = -torch.exp(self.A_log[direction_idx])  # (C_dim, d_state)
        D_param = self.D[direction_idx]  # (C_dim,)
        
        # Perform selective scan (token-by-token, no huge tensors)
        y_flat = selective_scan_1d(x_flat, delta, A, B_ssm, C_ssm, D_param, chunk_size=256)
        
        # Reshape back
        y = y_flat.transpose(1, 2).view(B, C_dim, H, W)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.proj_in(x)
        B, C, H, W = x.shape
        
        # Scan in 4 directions: left->right, right->left, top->bottom, bottom->top
        x_lr = x
        x_rl = torch.flip(x, dims=[-1])
        x_tb = x.transpose(-2, -1)
        x_bt = torch.flip(x.transpose(-2, -1), dims=[-1])
        
        directions = [x_lr, x_rl, x_tb, x_bt]
        outputs = []
        
        for i, x_dir in enumerate(directions):
            y_dir = self.scan_direction(x_dir, i)
            
            # Reverse transformations
            if i == 1:  # right->left
                y_dir = torch.flip(y_dir, dims=[-1])
            elif i == 2:  # top->bottom
                y_dir = y_dir.transpose(-2, -1)
            elif i == 3:  # bottom->top
                y_dir = torch.flip(y_dir, dims=[-1]).transpose(-2, -1)
            
            outputs.append(y_dir)
        
        # Fuse directions with learned weights
        weights = F.softmax(self.direction_weights, dim=0)
        fused = sum(w * out for w, out in zip(weights, outputs))
        
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
        d_state: int = 16,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.fuse_mode = fuse_mode

        self.axial_scan = SelectiveScan2D(channels, hidden_ratio, dropout, d_state)
        self.coronal_scan = SelectiveScan2D(channels, hidden_ratio, dropout, d_state)
        self.sagittal_scan = SelectiveScan2D(channels, hidden_ratio, dropout, d_state)

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


class SelectiveScan3D(nn.Module):
    """3D selective scan for true volumetric context modeling."""

    def __init__(
        self,
        channels: int,
        hidden_ratio: float = 1.0,
        dropout: float = 0.0,
        d_state: int = 16,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.d_state = d_state
        hidden_channels = max(8, int(math.ceil(channels * hidden_ratio)))
        
        # Input projection
        self.proj_in = nn.Conv3d(channels, hidden_channels, kernel_size=1, bias=False)
        
        # 8 scanning directions for 3D (all corners of a cube)
        # Directions: +++, ++-, +-+, +--, -++, -+-, --+, ---
        self.num_directions = 8
        
        # SSM parameters
        self.A_log = nn.Parameter(torch.randn(self.num_directions, hidden_channels, d_state))
        self.D = nn.Parameter(torch.ones(self.num_directions, hidden_channels))
        
        # Data-dependent parameter projections
        self.x_proj = nn.ModuleList([
            nn.Conv3d(hidden_channels, d_state + d_state + hidden_channels, kernel_size=1)
            for _ in range(self.num_directions)
        ])
        
        self.norm = nn.GroupNorm(num_groups=1, num_channels=hidden_channels, eps=eps)
        self.activation = nn.SiLU(inplace=True)
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        self.proj_out = nn.Conv3d(hidden_channels, channels, kernel_size=1, bias=False)
        
        # Learnable direction fusion
        self.direction_weights = nn.Parameter(torch.ones(self.num_directions) / self.num_directions)

    def scan_direction(self, x: torch.Tensor, direction_idx: int) -> torch.Tensor:
        """Scan in one 3D direction with SSM."""
        B, C_dim, D, H, W = x.shape
        
        # Flatten spatial dimensions for scanning (scan along D*H*W sequence)
        x_flat = x.flatten(2).transpose(1, 2)  # (B, D*H*W, C_dim)
        
        # Get data-dependent parameters
        params = self.x_proj[direction_idx](x)
        params_flat = params.flatten(2).transpose(1, 2)
        
        B_ssm, C_ssm, delta = params_flat.split([self.d_state, self.d_state, C_dim], dim=-1)
        delta = F.softplus(delta)
        
        # Get SSM matrices
        A = -torch.exp(self.A_log[direction_idx])
        D_param = self.D[direction_idx]
        
        # Perform selective scan (token-by-token, no huge tensors)
        y_flat = selective_scan_1d(x_flat, delta, A, B_ssm, C_ssm, D_param, chunk_size=256)
        
        # Reshape back
        y = y_flat.transpose(1, 2).view(B, C_dim, D, H, W)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.proj_in(x)
        B, C, D, H, W = x.shape
        
        # Define 8 3D scanning directions (all cube corners)
        directions = []
        for flip_d in [False, True]:
            for flip_h in [False, True]:
                for flip_w in [False, True]:
                    x_dir = x
                    if flip_d:
                        x_dir = torch.flip(x_dir, dims=[2])
                    if flip_h:
                        x_dir = torch.flip(x_dir, dims=[3])
                    if flip_w:
                        x_dir = torch.flip(x_dir, dims=[4])
                    directions.append((x_dir, flip_d, flip_h, flip_w))
        
        outputs = []
        for i, (x_dir, flip_d, flip_h, flip_w) in enumerate(directions):
            y_dir = self.scan_direction(x_dir, i)
            
            # Reverse transformations
            if flip_w:
                y_dir = torch.flip(y_dir, dims=[4])
            if flip_h:
                y_dir = torch.flip(y_dir, dims=[3])
            if flip_d:
                y_dir = torch.flip(y_dir, dims=[2])
            
            outputs.append(y_dir)
        
        # Fuse directions with learned weights
        weights = F.softmax(self.direction_weights, dim=0)
        fused = sum(w * out for w, out in zip(weights, outputs))
        
        fused = self.norm(fused)
        fused = self.activation(fused)
        fused = self.dropout(fused)
        fused = self.proj_out(fused)
        
        return fused + residual


class VMambaBlock3D(nn.Module):
    """3D VMamba block for volumetric global context."""

    def __init__(
        self,
        channels: int,
        hidden_ratio: float = 1.0,
        dropout: float = 0.0,
        use_se: bool = True,
        se_reduction: int = 8,
        d_state: int = 16,
    ) -> None:
        super().__init__()
        self.scan = SelectiveScan3D(channels, hidden_ratio, dropout, d_state)
        self.se = ChannelSE3D(channels, reduction=se_reduction) if use_se else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.scan(x)
        x = self.se(x)
        return x


__all__ = ["SelectiveScan2D", "SelectiveScan3D", "TriPlaneVMambaBlock", "VMambaBlock3D", "ChannelSE3D"]

