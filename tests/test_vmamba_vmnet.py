import torch

from nnunet.network_architecture.reproduction_mnet.mnet_vmamba import MNetVMamba
from nnunet.network_architecture.reproduction_mnet.vmamba_tri_plane import TriPlaneVMambaBlock


def _count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_tri_plane_vmamba_preserves_shape():
    block = TriPlaneVMambaBlock(channels=32, hidden_ratio=0.5, dropout=0.0)
    x = torch.randn(1, 32, 8, 32, 32)
    y = block(x)
    assert y.shape == x.shape


def test_mnet_vmamba_forward_and_budget():
    baseline = MNetVMamba(1, 3, vmamba_cfg={"enabled": False})
    base_params = _count_params(baseline)

    cfg = {
        "enabled": True,
        "down_stages": [2],
        "up_stages": [2],
        "bottleneck_stages": [4],
        "hidden_ratio": 0.25,
        "dropout": 0.0,
        "fuse_mode": "concat",
        "use_se": True,
    }
    hybrid = MNetVMamba(1, 3, vmamba_cfg=cfg)
    hybrid_params = _count_params(hybrid)

    assert hybrid_params <= base_params * 1.1

    x = torch.randn(1, 1, 8, 64, 64)
    with torch.no_grad():
        out = hybrid(x)
    assert isinstance(out, tuple)
    assert out[0].shape[-3:] == (8, 64, 64)

