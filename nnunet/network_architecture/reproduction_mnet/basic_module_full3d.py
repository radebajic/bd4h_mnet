from torch import nn
import torch
import warnings
from typing import Optional

# Import the full 3D VMamba block from vmamba_tri_plane
try:
    from nnunet.network_architecture.reproduction_mnet.vmamba_tri_plane import VMambaBlock3D
    VMAMBA_3D_AVAILABLE = True
except ImportError:
    VMambaBlock3D = None
    VMAMBA_3D_AVAILABLE = False

# Re-export base classes from basic_module for compatibility
from nnunet.network_architecture.reproduction_mnet.basic_module import (
    CNA3d, CB3d, CB3dSeparable, BasicNet, ZScan, CBzMamba, 
    MAMBA1D_KWARG_CANDIDATES, _MAMBA_FALLBACK_WARNED
)

__all__ = [
    'CNA3d', 'CB3d', 'CB3dSeparable', 'BasicNet', 'ZScan', 'CBzMamba',
    'CB3dFull3DVMamba', 'VMAMBA_3D_AVAILABLE'
]


class CB3dFull3DVMamba(nn.Module):
    """
    Full 3D VMamba block that flattens D×H×W to sequence L=D*H*W.
    This is the "true" 3D Mamba approach with residual connection.
    
    Architecture:
        x → 1×1×1 Conv (projection) → Norm → Act
          → VMambaBlock3D (flattens to L=D*H*W, applies selective scan)
          → + residual(x)
    
    Args:
        in_channels: Input channels
        out_channels: Output channels  
        hidden_ratio: Hidden dimension multiplier for VMamba (default: 1.0)
        d_state: SSM state dimension (default: 16)
        dropout: Dropout rate (default: 0.0)
        use_se: Whether to use channel SE attention (default: True)
        se_reduction: SE reduction ratio (default: 8)
        norm_kwargs: Kwargs for InstanceNorm3d
        act_kwargs: Kwargs for LeakyReLU
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        hidden_ratio: float = 1.0,
        d_state: int = 16,
        dropout: float = 0.0,
        use_se: bool = True,
        se_reduction: int = 8,
        norm_kwargs: dict = None,
        act_kwargs: dict = None
    ):
        super().__init__()
        
        if norm_kwargs is None:
            norm_kwargs = {'affine': True}
        if act_kwargs is None:
            act_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        
        self.vmamba_available = VMAMBA_3D_AVAILABLE
        
        if not self.vmamba_available:
            warnings.warn(
                "VMambaBlock3D not available. Falling back to standard CB3d. "
                "Check that vmamba_tri_plane.py is importable.",
                RuntimeWarning,
                stacklevel=2
            )
            # Fallback to standard convolutions
            self.fallback = CB3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kSize=(3, 3),
                stride=(1, 1),
                padding=(1, 1, 1),
                norm_args=(norm_kwargs, norm_kwargs),
                activation_args=(act_kwargs, act_kwargs)
            )
            self.pre = None
            self.vmamba = None
            self.post = None
            return
        
        # Pre-projection to match input/output channels
        self.pre = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.n1 = nn.InstanceNorm3d(out_channels, **norm_kwargs)
        self.a1 = nn.LeakyReLU(**act_kwargs)
        
        # Full 3D VMamba block
        self.vmamba = VMambaBlock3D(
            channels=out_channels,
            hidden_ratio=hidden_ratio,
            dropout=dropout,
            use_se=use_se,
            se_reduction=se_reduction,
            d_state=d_state
        )
        
        # No post projection needed since VMamba preserves channels
        self.post = None
        self.fallback = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.
        
        Args:
            x: Input tensor (B, C_in, D, H, W)
            
        Returns:
            Output tensor (B, C_out, D, H, W)
        """
        if not self.vmamba_available:
            return self.fallback(x)
        
        # Project and normalize
        residual = self.pre(x)
        x = self.a1(self.n1(residual))
        
        # Apply full 3D VMamba (internally flattens to L=D*H*W)
        x = self.vmamba(x)
        
        # Residual connection
        return x + residual


if __name__ == '__main__':
    # Test the module
    print(f"VMambaBlock3D available: {VMAMBA_3D_AVAILABLE}")
    
    net = CB3dFull3DVMamba(
        in_channels=64,
        out_channels=64,
        hidden_ratio=1.0,
        d_state=16,
        dropout=0.1,
        use_se=True,
        se_reduction=8
    )
    
    x = torch.randn(1, 64, 16, 32, 32)
    y = net(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Parameters: {sum(p.numel() for p in net.parameters()):,}")



