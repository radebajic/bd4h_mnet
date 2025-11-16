"""
Trainer for MNet with Full 3D VMamba support.

This is a minimal extension of myTrainer_reproduction_WandB that:
1. Uses MNetFull3D instead of MNet
2. Adds Full3D VMamba-specific parameters (only the new ones)
3. Keeps ALL other settings identical to the parent trainer

The only differences:
- initialize_network() uses MNetFull3D
- Adds vmamba_* parameters for Full3D configuration
"""

from typing import Optional
import torch
from nnunet.training.network_training.myTrainer_reproduction_WandB import myTrainer_reproduction_WandB
from nnunet.network_architecture.reproduction_mnet.mnet_full3d import MNetFull3D
from nnunet.utilities.nd_softmax import softmax_helper


class myTrainer_reproduction_WandB_Full3D(myTrainer_reproduction_WandB):
    """
    Extends myTrainer_reproduction_WandB to use MNetFull3D with full volumetric VMamba.
    
    All settings identical to parent except:
    - Uses MNetFull3D network
    - Adds vmamba_* parameters for Full3D-specific configuration
    """
    
    def __init__(
        self,
        plans_file,
        fold,
        output_folder=None,
        dataset_directory=None,
        batch_dice=True,
        stage=None,
        unpack_data=True,
        deterministic=True,
        fp16=False,
        *,
        max_num_epochs: int = 1,
        gated_fusion: str = "spatial",
        width_mult: float = 1.0,
        use_sep3d: bool = False,
        use_checkpoint: bool = False,
        cat_reduce: bool = False,
    ):
        # Call parent __init__ to inherit ALL existing functionality
        super().__init__(
            plans_file, fold, output_folder, dataset_directory, batch_dice, stage,
            unpack_data, deterministic, fp16,
            max_num_epochs=max_num_epochs,
            gated_fusion=gated_fusion,
            width_mult=width_mult,
            use_sep3d=use_sep3d,
            use_checkpoint=use_checkpoint,
            cat_reduce=cat_reduce
        )
        
        # Only add NEW parameters specific to Full3D VMamba
        # (Hydra will override these after construction)
        self.vmamba_hidden_ratio = 1.0
        self.vmamba_d_state = 16
        self.vmamba_dropout = 0.0
        self.vmamba_use_se = True
        self.vmamba_se_reduction = 8
        
        self.print_to_log_file("Using Full3D VMamba trainer (flattens D×H×W to sequence)")

    def initialize_network(self):
        """
        Initialize MNetFull3D instead of MNet.
        All other parameters identical to parent.
        """
        self.network = MNetFull3D(
            self.num_input_channels, self.num_classes,
            kn=(32, 48, 64, 80, 96),
            ds=True, FMU='sub',
            width_mult=getattr(self, "width_mult", 1.0),
            use_sep3d=getattr(self, "use_sep3d", False),
            use_checkpoint=getattr(self, "use_checkpoint", False),
            cat_reduce=getattr(self, "cat_reduce", False),
            gated_fusion=getattr(self, "gated_fusion", None),
            # Original VMamba stage controls (kept identical)
            vm_down_stages=getattr(self, "vm_down_stages", []),
            vm_up_stages=getattr(self, "vm_up_stages", []),
            vm_bottleneck_stages=getattr(self, "vm_bottleneck_stages", []),
            # NEW: Full3D VMamba-specific parameters
            vmamba_hidden_ratio=getattr(self, "vmamba_hidden_ratio", 1.0),
            vmamba_d_state=getattr(self, "vmamba_d_state", 16),
            vmamba_dropout=getattr(self, "vmamba_dropout", 0.0),
            vmamba_use_se=getattr(self, "vmamba_use_se", True),
            vmamba_se_reduction=getattr(self, "vmamba_se_reduction", 8),
        )
        
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

        total_params = sum(p.numel() for p in self.network.parameters())
        trainable_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        self.print_to_log_file(f"Model params: total={total_params:,}, trainable={trainable_params:,}")

        if hasattr(self.network, "save_arch_summary"):
            try:
                self.network.save_arch_summary(self.output_folder)
            except Exception as exc:
                self.print_to_log_file(f"Arch summary save failed: {exc}")

    def _get_baseline_param_count(self) -> Optional[int]:
        """Override to use MNetFull3D for baseline comparison."""
        if self._baseline_param_count is not None:
            return self._baseline_param_count

        baseline_net = None
        try:
            import contextlib
            import io
            with torch.no_grad():
                buffer = io.StringIO()
                with contextlib.redirect_stdout(buffer):
                    baseline_net = MNetFull3D(
                        self.num_input_channels,
                        self.num_classes,
                        kn=(32, 48, 64, 80, 96),
                        ds=True,
                        FMU='sub',
                        width_mult=getattr(self, "width_mult", 1.0),
                        use_sep3d=getattr(self, "use_sep3d", False),
                        use_checkpoint=getattr(self, "use_checkpoint", False),
                        cat_reduce=getattr(self, "cat_reduce", False),
                        gated_fusion=getattr(self, "gated_fusion", None),
                        vm_down_stages=[],
                        vm_up_stages=[],
                        vm_bottleneck_stages=[],
                        vmamba_hidden_ratio=getattr(self, "vmamba_hidden_ratio", 1.0),
                        vmamba_d_state=getattr(self, "vmamba_d_state", 16),
                        vmamba_dropout=0.0,
                        vmamba_use_se=getattr(self, "vmamba_use_se", True),
                        vmamba_se_reduction=getattr(self, "vmamba_se_reduction", 8),
                    )
                params = sum(p.numel() for p in baseline_net.parameters())
            self._baseline_param_count = params
        except Exception as exc:
            self.print_to_log_file(f"Baseline param count computation failed: {exc}")
            self._baseline_param_count = None
        finally:
            if baseline_net is not None:
                del baseline_net

        return self._baseline_param_count







