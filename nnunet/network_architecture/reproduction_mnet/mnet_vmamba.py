from typing import Optional, Dict, Set

import torch
import torch.nn.functional as F
from torch import nn

from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.network_architecture.reproduction_mnet.basic_module import CB3d, BasicNet
from nnunet.network_architecture.reproduction_mnet.vmamba_tri_plane import TriPlaneVMambaBlock


def FMU(x1: torch.Tensor, x2: torch.Tensor, mode: str = 'sub') -> torch.Tensor:
    if mode == 'sum':
        return torch.add(x1, x2)
    if mode == 'sub':
        return torch.abs(x1 - x2)
    if mode == 'cat':
        return torch.cat((x1, x2), dim=1)
    raise ValueError(f'Unexpected FMU mode: {mode}')


class DownVM(BasicNet):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mode: tuple,
        FMU_mode: str = 'sub',
        downsample: bool = True,
        min_z: int = 8,
        use_vmamba: bool = False,
        vmamba_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.mode_in, self.mode_out = mode
        self.downsample = downsample
        self.FMU = FMU_mode
        self.min_z = min_z

        norm_args = (self.norm_kwargs, self.norm_kwargs)
        activation_args = (self.activation_kwargs, self.activation_kwargs)

        self.vmamba_enabled = use_vmamba
        self.vmamba_kwargs = vmamba_kwargs or {}
        if self.vmamba_enabled:
            self.vm_proj = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
            self.vm_branch = TriPlaneVMambaBlock(out_channels, **self.vmamba_kwargs)
            self.fuse2d = self._make_fuse_layer(out_channels) if self.mode_out in ('2d', 'both') else None
            self.fuse3d = self._make_fuse_layer(out_channels) if self.mode_out in ('3d', 'both') else None
        else:
            self.vm_proj = None
            self.vm_branch = None
            self.fuse2d = None
            self.fuse3d = None

        if self.mode_out in ('2d', 'both'):
            self.CB2d = CB3d(in_channels=in_channels, out_channels=out_channels,
                             kSize=((1, 3, 3), (1, 3, 3)), stride=(1, 1), padding=(0, 1, 1),
                             norm_args=norm_args, activation_args=activation_args)

        if self.mode_out in ('3d', 'both'):
            self.CB3d = CB3d(in_channels=in_channels, out_channels=out_channels,
                             kSize=(3, 3), stride=(1, 1), padding=(1, 1, 1),
                             norm_args=norm_args, activation_args=activation_args)

    @staticmethod
    def _make_fuse_layer(channels: int) -> nn.Sequential:
        """Create gated fusion layer for better feature integration."""
        return nn.Sequential(
            nn.Conv3d(channels * 2, channels * 2, kernel_size=1, bias=False),
            nn.InstanceNorm3d(channels * 2, affine=True),
            nn.SiLU(inplace=True),
            nn.Conv3d(channels * 2, channels, kernel_size=1, bias=True),
        )

    @staticmethod
    def _fuse(local: torch.Tensor, vm_feat: Optional[torch.Tensor], fuse_layer: Optional[nn.Module]) -> torch.Tensor:
        """Gated fusion: learns to balance local and global features."""
        if vm_feat is None or fuse_layer is None:
            return local
        
        # Compute gating signal
        combined = torch.cat([local, vm_feat], dim=1)
        gate = torch.sigmoid(fuse_layer(combined))
        
        # Gated combination with residual
        return gate * vm_feat + (1 - gate) * local

    def forward(self, x):
        if self.downsample:
            if self.mode_in == 'both':
                x2d, x3d = x
                p2d = F.max_pool3d(x2d, kernel_size=(1, 2, 2), stride=(1, 2, 2))
                if x3d.shape[2] >= self.min_z:
                    p3d = F.max_pool3d(x3d, kernel_size=(2, 2, 2), stride=(2, 2, 2))
                else:
                    p3d = F.max_pool3d(x3d, kernel_size=(1, 2, 2), stride=(1, 2, 2))
                x = FMU(p2d, p3d, mode=self.FMU)
            elif self.mode_in == '2d':
                x = F.max_pool3d(x, kernel_size=(1, 2, 2), stride=(1, 2, 2))
            elif self.mode_in == '3d':
                if x.shape[2] >= self.min_z:
                    x = F.max_pool3d(x, kernel_size=(2, 2, 2), stride=(2, 2, 2))
                else:
                    x = F.max_pool3d(x, kernel_size=(1, 2, 2), stride=(1, 2, 2))

        vm_feat = None
        if self.vmamba_enabled:
            projected = self.vm_proj(x)
            vm_feat = self.vm_branch(projected)

        if self.mode_out == '2d':
            return self._fuse(self.CB2d(x), vm_feat, self.fuse2d)
        if self.mode_out == '3d':
            return self._fuse(self.CB3d(x), vm_feat, self.fuse3d)
        out2d = self._fuse(self.CB2d(x), vm_feat, self.fuse2d)
        out3d = self._fuse(self.CB3d(x), vm_feat, self.fuse3d)
        return out2d, out3d


class UpVM(BasicNet):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mode: tuple,
        FMU_mode: str = 'sub',
        use_vmamba: bool = False,
        vmamba_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.mode_in, self.mode_out = mode
        self.FMU = FMU_mode

        norm_args = (self.norm_kwargs, self.norm_kwargs)
        activation_args = (self.activation_kwargs, self.activation_kwargs)

        self.vmamba_enabled = use_vmamba
        self.vmamba_kwargs = vmamba_kwargs or {}
        if self.vmamba_enabled:
            self.vm_branch = TriPlaneVMambaBlock(in_channels, **self.vmamba_kwargs)
            self.vm_proj2d = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False) if self.mode_out in ('2d', 'both') else None
            self.vm_proj3d = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False) if self.mode_out in ('3d', 'both') else None
            self.fuse2d = self._make_fuse_layer(out_channels) if self.mode_out in ('2d', 'both') else None
            self.fuse3d = self._make_fuse_layer(out_channels) if self.mode_out in ('3d', 'both') else None
        else:
            self.vm_branch = None
            self.vm_proj2d = None
            self.vm_proj3d = None
            self.fuse2d = None
            self.fuse3d = None

        if self.mode_out in ('2d', 'both'):
            self.CB2d = CB3d(in_channels=in_channels, out_channels=out_channels,
                             kSize=((1, 3, 3), (1, 3, 3)), stride=(1, 1), padding=(0, 1, 1),
                             norm_args=norm_args, activation_args=activation_args)

        if self.mode_out in ('3d', 'both'):
            self.CB3d = CB3d(in_channels=in_channels, out_channels=out_channels,
                             kSize=(3, 3), stride=(1, 1), padding=(1, 1, 1),
                             norm_args=norm_args, activation_args=activation_args)

    @staticmethod
    def _make_fuse_layer(channels: int) -> nn.Sequential:
        """Create gated fusion layer for better feature integration."""
        return nn.Sequential(
            nn.Conv3d(channels * 2, channels * 2, kernel_size=1, bias=False),
            nn.InstanceNorm3d(channels * 2, affine=True),
            nn.SiLU(inplace=True),
            nn.Conv3d(channels * 2, channels, kernel_size=1, bias=True),
        )

    @staticmethod
    def _fuse(local: torch.Tensor, vm_feat: Optional[torch.Tensor], proj: Optional[nn.Module], fuse_layer: Optional[nn.Module]) -> torch.Tensor:
        """Gated fusion: learns to balance local and global features."""
        if vm_feat is None or proj is None or fuse_layer is None:
            return local
        
        vm_proj = proj(vm_feat)
        combined = torch.cat([local, vm_proj], dim=1)
        gate = torch.sigmoid(fuse_layer(combined))
        
        # Gated combination with residual
        return gate * vm_proj + (1 - gate) * local

    def forward(self, x):
        x2d, xskip2d, x3d, xskip3d = x

        target_size = xskip2d.shape[2:]
        up2d = F.interpolate(x2d, size=target_size, mode='trilinear', align_corners=False)
        up3d = F.interpolate(x3d, size=target_size, mode='trilinear', align_corners=False)

        cat = torch.cat([FMU(xskip2d, xskip3d, self.FMU), FMU(up2d, up3d, self.FMU)], dim=1)

        vm_feat = self.vm_branch(cat) if self.vmamba_enabled else None

        if self.mode_out == '2d':
            return self._fuse(self.CB2d(cat), vm_feat, self.vm_proj2d, self.fuse2d)
        if self.mode_out == '3d':
            return self._fuse(self.CB3d(cat), vm_feat, self.vm_proj3d, self.fuse3d)
        out2d = self._fuse(self.CB2d(cat), vm_feat, self.vm_proj2d, self.fuse2d)
        out3d = self._fuse(self.CB3d(cat), vm_feat, self.vm_proj3d, self.fuse3d)
        return out2d, out3d


class MNetVMamba(SegmentationNetwork):
    DEFAULT_BATCH_SIZE_3D = 2
    DEFAULT_PATCH_SIZE_3D = (64, 192, 160)
    SPACING_FACTOR_BETWEEN_STAGES = 2
    BASE_NUM_FEATURES_3D = 30
    MAX_NUMPOOL_3D = 999
    MAX_NUM_FILTERS_3D = 320

    DEFAULT_PATCH_SIZE_2D = (256, 256)
    BASE_NUM_FEATURES_2D = 30
    DEFAULT_BATCH_SIZE_2D = 50
    MAX_NUMPOOL_2D = 999
    MAX_FILTERS_2D = 480

    use_this_for_batch_size_computation_2D = 19739648
    use_this_for_batch_size_computation_3D = 520000000

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        kn=(32, 48, 64, 80, 96),
        ds: bool = True,
        FMU_mode: str = 'sub',
        vmamba_cfg: Optional[Dict] = None,
    ) -> None:
        super().__init__()
        self.conv_op = nn.Conv3d
        self._deep_supervision = self.do_ds = ds
        self.num_classes = num_classes

        channel_factor = {'sum': 1, 'sub': 1, 'cat': 2}
        fct = channel_factor[FMU_mode]

        self.vmamba_cfg = vmamba_cfg or {}
        self.vm_kwargs = {
            'hidden_ratio': self.vmamba_cfg.get('hidden_ratio', self.vmamba_cfg.get('axial_reduce', 1.0)),  # Increased from 0.5
            'dropout': self.vmamba_cfg.get('dropout', 0.1),  # Added dropout for regularization
            'fuse_mode': self.vmamba_cfg.get('fuse_mode', 'concat'),
            'use_se': self.vmamba_cfg.get('use_se', True),
            'se_reduction': self.vmamba_cfg.get('se_reduction', 8),
            'd_state': self.vmamba_cfg.get('d_state', 16),  # SSM state dimension
        }

        vm_down = {int(v) for v in self.vmamba_cfg.get('down_stages', [])}
        vm_up = {int(v) for v in self.vmamba_cfg.get('up_stages', [])}
        vm_bn = {int(v) for v in self.vmamba_cfg.get('bottleneck_stages', [])}
        vm_targets = set(self.vmamba_cfg.get('targets', []))
        backend_available = self.vmamba_cfg.get('backend_available', True)

        vm_enabled = bool(self.vmamba_cfg.get('enabled', True))

        def _use(block_name: str, stage: str, stage_idx: int) -> bool:
            if not vm_enabled:
                return False
            if not backend_available:
                return False
            if block_name in vm_targets:
                return True
            if stage == 'down':
                return stage_idx in vm_down or self.vmamba_cfg.get('in_down', False)
            if stage == 'up':
                return stage_idx in vm_up or self.vmamba_cfg.get('in_up', False)
            if stage == 'bottleneck':
                return stage_idx in vm_bn or self.vmamba_cfg.get('in_bottleneck', False)
            return False

        # Level 1
        self.down11 = DownVM(in_channels, kn[0], ('/', 'both'), downsample=False,
                              use_vmamba=_use('down11', 'down', 1), vmamba_kwargs=self.vm_kwargs)
        self.down12 = DownVM(kn[0], kn[1], ('2d', 'both'), FMU_mode,
                              use_vmamba=_use('down12', 'down', 1), vmamba_kwargs=self.vm_kwargs)
        self.down13 = DownVM(kn[1], kn[2], ('2d', 'both'), FMU_mode,
                              use_vmamba=_use('down13', 'down', 1), vmamba_kwargs=self.vm_kwargs)
        self.down14 = DownVM(kn[2], kn[3], ('2d', 'both'), FMU_mode,
                              use_vmamba=_use('down14', 'down', 1), vmamba_kwargs=self.vm_kwargs)
        self.bottleneck1 = DownVM(kn[3], kn[4], ('2d', '2d'), FMU_mode,
                                   use_vmamba=_use('bottleneck1', 'bottleneck', 1), vmamba_kwargs=self.vm_kwargs)
        self.up11 = UpVM(fct * (kn[3] + kn[4]), kn[3], ('both', '2d'), FMU_mode,
                          use_vmamba=_use('up11', 'up', 1), vmamba_kwargs=self.vm_kwargs)
        self.up12 = UpVM(fct * (kn[2] + kn[3]), kn[2], ('both', '2d'), FMU_mode,
                          use_vmamba=_use('up12', 'up', 1), vmamba_kwargs=self.vm_kwargs)
        self.up13 = UpVM(fct * (kn[1] + kn[2]), kn[1], ('both', '2d'), FMU_mode,
                          use_vmamba=_use('up13', 'up', 1), vmamba_kwargs=self.vm_kwargs)
        self.up14 = UpVM(fct * (kn[0] + kn[1]), kn[0], ('both', 'both'), FMU_mode,
                          use_vmamba=_use('up14', 'up', 1), vmamba_kwargs=self.vm_kwargs)

        # Level 2
        self.down21 = DownVM(kn[0], kn[1], ('3d', 'both'), FMU_mode,
                              use_vmamba=_use('down21', 'down', 2), vmamba_kwargs=self.vm_kwargs)
        self.down22 = DownVM(fct * kn[1], kn[2], ('both', 'both'), FMU_mode,
                              use_vmamba=_use('down22', 'down', 2), vmamba_kwargs=self.vm_kwargs)
        self.down23 = DownVM(fct * kn[2], kn[3], ('both', 'both'), FMU_mode,
                              use_vmamba=_use('down23', 'down', 2), vmamba_kwargs=self.vm_kwargs)
        self.bottleneck2 = DownVM(fct * kn[3], kn[4], ('both', 'both'), FMU_mode,
                                   use_vmamba=_use('bottleneck2', 'bottleneck', 2), vmamba_kwargs=self.vm_kwargs)
        self.up21 = UpVM(fct * (kn[3] + kn[4]), kn[3], ('both', 'both'), FMU_mode,
                          use_vmamba=_use('up21', 'up', 2), vmamba_kwargs=self.vm_kwargs)
        self.up22 = UpVM(fct * (kn[2] + kn[3]), kn[2], ('both', 'both'), FMU_mode,
                          use_vmamba=_use('up22', 'up', 2), vmamba_kwargs=self.vm_kwargs)
        self.up23 = UpVM(fct * (kn[1] + kn[2]), kn[1], ('both', '3d'), FMU_mode,
                          use_vmamba=_use('up23', 'up', 2), vmamba_kwargs=self.vm_kwargs)

        # Level 3
        self.down31 = DownVM(kn[1], kn[2], ('3d', 'both'), FMU_mode,
                              use_vmamba=_use('down31', 'down', 3), vmamba_kwargs=self.vm_kwargs)
        self.down32 = DownVM(fct * kn[2], kn[3], ('both', 'both'), FMU_mode,
                              use_vmamba=_use('down32', 'down', 3), vmamba_kwargs=self.vm_kwargs)
        self.bottleneck3 = DownVM(fct * kn[3], kn[4], ('both', 'both'), FMU_mode,
                                   use_vmamba=_use('bottleneck3', 'bottleneck', 3), vmamba_kwargs=self.vm_kwargs)
        self.up31 = UpVM(fct * (kn[3] + kn[4]), kn[3], ('both', 'both'), FMU_mode,
                          use_vmamba=_use('up31', 'up', 3), vmamba_kwargs=self.vm_kwargs)
        self.up32 = UpVM(fct * (kn[2] + kn[3]), kn[2], ('both', '3d'), FMU_mode,
                          use_vmamba=_use('up32', 'up', 3), vmamba_kwargs=self.vm_kwargs)

        # Level 4 & deepest bottleneck
        self.down41 = DownVM(kn[2], kn[3], ('3d', 'both'), FMU_mode,
                              use_vmamba=_use('down41', 'down', 4), vmamba_kwargs=self.vm_kwargs)
        self.bottleneck4 = DownVM(fct * kn[3], kn[4], ('both', 'both'), FMU_mode,
                                   use_vmamba=_use('bottleneck4', 'bottleneck', 4), vmamba_kwargs=self.vm_kwargs)
        self.up41 = UpVM(fct * (kn[3] + kn[4]), kn[3], ('both', '3d'), FMU_mode,
                          use_vmamba=_use('up41', 'up', 4), vmamba_kwargs=self.vm_kwargs)

        self.bottleneck5 = DownVM(kn[3], kn[4], ('3d', '3d'), FMU_mode,
                                   use_vmamba=_use('bottleneck5', 'bottleneck', 5), vmamba_kwargs=self.vm_kwargs)

        self.outputs = nn.ModuleList(
            [nn.Conv3d(c, num_classes, kernel_size=1, stride=1, padding=0, bias=False)
             for c in [kn[0], kn[1], kn[1], kn[2], kn[2], kn[3], kn[3]]]
        )

    def forward(self, x: torch.Tensor):
        down11 = self.down11(x)
        down12 = self.down12(down11[0])
        down13 = self.down13(down12[0])
        down14 = self.down14(down13[0])
        bottleneck1 = self.bottleneck1(down14[0])

        down21 = self.down21(down11[1])
        down22 = self.down22([down21[0], down12[1]])
        down23 = self.down23([down22[0], down13[1]])
        bottleneck2 = self.bottleneck2([down23[0], down14[1]])

        down31 = self.down31(down21[1])
        down32 = self.down32([down31[0], down22[1]])
        bottleneck3 = self.bottleneck3([down32[0], down23[1]])

        down41 = self.down41(down31[1])
        bottleneck4 = self.bottleneck4([down41[0], down32[1]])
        bottleneck5 = self.bottleneck5(down41[1])

        up41 = self.up41([bottleneck4[0], down41[0], bottleneck5, down41[1]])

        up31 = self.up31([bottleneck3[0], down32[0], bottleneck4[1], down32[1]])
        up32 = self.up32([up31[0], down31[0], up41, down31[1]])

        up21 = self.up21([bottleneck2[0], down23[0], bottleneck3[1], down23[1]])
        up22 = self.up22([up21[0], down22[0], up31[1], down22[1]])
        up23 = self.up23([up22[0], down21[0], up32, down21[1]])

        up11 = self.up11([bottleneck1, down14[0], bottleneck2[1], down14[1]])
        up12 = self.up12([up11, down13[0], up21[1], down13[1]])
        up13 = self.up13([up12, down12[0], up22[1], down12[1]])
        up14 = self.up14([up13, down11[0], up23, down11[1]])

        if self._deep_supervision and self.do_ds:
            features = [up14[0] + up14[1], up23, up13, up32, up12, up41, up11]
            return tuple(self.outputs[i](features[i]) for i in range(7))
        return self.outputs[0](up14[0] + up14[1])


__all__ = ["MNetVMamba", "DownVM", "UpVM"]

