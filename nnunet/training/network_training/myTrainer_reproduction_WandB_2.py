import contextlib
import io
import os
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Tuple, Sequence

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from batchgenerators.utilities.file_and_folder_operations import *  # noqa: F401,F403
from omegaconf import DictConfig
from sklearn.model_selection import KFold
from torch import nn
from torch.cuda.amp import autocast

from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.network_architecture.reproduction_mnet.mnet_vmamba import MNetVMamba
from nnunet.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
from nnunet.training.data_augmentation.default_data_augmentation import (
    default_2D_augmentation_params,
    default_3D_augmentation_params,
    get_patch_size,
)
from nnunet.training.dataloading.dataset_loading import unpack_dataset
from nnunet.training.learning_rate.poly_lr import poly_lr
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda

try:
    import wandb
except Exception:  # pragma: no cover
    wandb = None


class myTrainer_reproduction_WandB_2(nnUNetTrainer):
    """Trainer variant that wires the VMamba-augmented reproduction MNet."""

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
        width_mult: float = 1.0,
        gated_fusion: Optional[str] = None,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)

        self.max_num_epochs = max_num_epochs
        self.initial_lr = 1e-2
        self.deep_supervision_scales: Optional[Sequence[Sequence[float]]] = None
        self.ds_loss_weights: Optional[np.ndarray] = None
        self.pin_memory = True

        # architecture toggles (mostly for logging/backwards compatibility)
        self.width_mult = width_mult
        self.gated_fusion = gated_fusion
        self.use_checkpoint = use_checkpoint

        # VMamba controls (Hydra/CLI can overwrite later)
        self.vmamba_enabled = True
        self.vm_down_stages = []
        self.vm_up_stages = []
        self.vm_bottleneck_stages = []
        self.vmamba_hidden_ratio = 0.5
        self.vmamba_dropout = 0.0
        self.vmamba_fuse_mode = "concat"
        self.vmamba_use_se = True
        self.vmamba_backend_available = True
        self.vmamba_targets = []

        # WandB defaults
        self.wandb_enabled: bool = False
        self.wandb_project: Optional[str] = None
        self.wandb_run_name: Optional[str] = None
        self.wandb_group: Optional[str] = None
        self._wb_run = None

        self._global_step = 0
        self._ema_loss: Optional[float] = None
        self._epoch_loss_sum = 0.0
        self._epoch_loss_cnt = 0
        self._baseline_param_count: Optional[int] = None
        self.amp_grad_scaler = None

        cudnn.benchmark = True
        self.print_to_log_file(f"Trainer2 config: max_epochs={self.max_num_epochs}, vmamba_enabled={self.vmamba_enabled}")

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _vmamba_cfg(self) -> dict:
        return {
            "enabled": bool(self.vmamba_enabled),
            "down_stages": list(getattr(self, "vm_down_stages", [])),
            "up_stages": list(getattr(self, "vm_up_stages", [])),
            "bottleneck_stages": list(getattr(self, "vm_bottleneck_stages", [])),
            "targets": list(getattr(self, "vmamba_targets", [])),
            "hidden_ratio": float(getattr(self, "vmamba_hidden_ratio", 0.5)),
            "dropout": float(getattr(self, "vmamba_dropout", 0.0)),
            "fuse_mode": getattr(self, "vmamba_fuse_mode", "concat"),
            "use_se": bool(getattr(self, "vmamba_use_se", True)),
            "backend_available": bool(getattr(self, "vmamba_backend_available", True)),
        }

    def _baseline_vmamba_cfg(self) -> dict:
        cfg = self._vmamba_cfg()
        cfg.update({
            "enabled": False,
            "down_stages": [],
            "up_stages": [],
            "bottleneck_stages": [],
            "targets": [],
        })
        return cfg

    def _wandb_log_architecture(self) -> None:
        if wandb is None or getattr(self, "_wb_run", None) is None or not hasattr(self, "network"):
            return

        arch_cfg = {
            "arch/network": self.network.__class__.__name__,
            "arch/vmamba/enabled": bool(getattr(self, "vmamba_enabled", False)),
            "arch/vmamba/down_stages": list(getattr(self, "vm_down_stages", [])),
            "arch/vmamba/up_stages": list(getattr(self, "vm_up_stages", [])),
            "arch/vmamba/bottleneck_stages": list(getattr(self, "vm_bottleneck_stages", [])),
            "arch/vmamba/hidden_ratio": getattr(self, "vmamba_hidden_ratio", None),
            "arch/vmamba/dropout": getattr(self, "vmamba_dropout", None),
            "arch/vmamba/fuse_mode": getattr(self, "vmamba_fuse_mode", None),
            "arch/vmamba/use_se": getattr(self, "vmamba_use_se", None),
        }

        try:
            total_params = sum(p.numel() for p in self.network.parameters())
            trainable_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
            arch_cfg["arch/params_total_m"] = round(total_params / 1e6, 3)
            arch_cfg["arch/params_trainable_m"] = round(trainable_params / 1e6, 3)
        except Exception:
            pass

        baseline = self._get_baseline_param_count()
        if baseline is not None and "arch/params_total_m" in arch_cfg:
            total = arch_cfg["arch/params_total_m"] * 1e6
            delta = total - baseline
            arch_cfg["arch/params_delta_m"] = round(delta / 1e6, 3)
            arch_cfg["arch/params_delta_pct"] = round((delta / baseline) * 100, 3)

        try:
            wandb.config.update(arch_cfg, allow_val_change=True)
            wandb.log({k: v for k, v in arch_cfg.items() if isinstance(v, (int, float, str))}, step=getattr(self, "epoch", 0))
        except Exception:
            pass

        try:
            arch_dir = Path(self.output_folder)
            arch_dir.mkdir(parents=True, exist_ok=True)
            arch_txt = arch_dir / "architecture_vmamba.txt"
            with open(arch_txt, "w", encoding="utf-8") as f:
                f.write(str(self.network))
            artifact = wandb.Artifact("architecture_vmamba", type="text")
            artifact.add_file(str(arch_txt), name="architecture_vmamba.txt")
            self._wb_run.log_artifact(artifact)
        except Exception:
            pass

    def _get_baseline_param_count(self) -> Optional[int]:
        if self._baseline_param_count is not None:
            return self._baseline_param_count
        baseline_net = None
        try:
            with torch.no_grad():
                buffer = io.StringIO()
                with contextlib.redirect_stdout(buffer):
                    baseline_net = MNetVMamba(
                        self.num_input_channels,
                        self.num_classes,
                        kn=(32, 48, 64, 80, 96),
                        ds=True,
                        FMU_mode='sub',
                        vmamba_cfg=self._baseline_vmamba_cfg(),
                    )
                params = sum(p.numel() for p in baseline_net.parameters())
                self._baseline_param_count = params
        except Exception as exc:
            self.print_to_log_file(f"Baseline param count failed: {exc}")
            self._baseline_param_count = None
        finally:
            if baseline_net is not None:
                del baseline_net
        return self._baseline_param_count

    # ------------------------------------------------------------------
    # nnUNetTrainer overrides
    # ------------------------------------------------------------------
    def initialize(self, training: bool = True, force_load_plans: bool = False):
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)
            if force_load_plans or (self.plans is None):
                self.load_plans_file()

            self.process_plans(self.plans)
            self.setup_DA_params()

            z = self.patch_size[0]
            if z >= 32:
                z_scale = [0.5, 0.25, 0.125]
            elif z >= 16:
                z_scale = [0.5, 0.25, 0.25]
            elif z >= 8:
                z_scale = [0.5, 0.5, 0.5]
            else:
                z_scale = [1, 1, 1]

            self.deep_supervision_scales = [
                [1, 1, 1],
                [z_scale[0], 0.5, 0.5],
                [1, 0.5, 0.5],
                [z_scale[1], 0.25, 0.25],
                [1, 0.25, 0.25],
                [z_scale[2], 0.125, 0.125],
                [1, 0.125, 0.125],
            ]
            self.ds_loss_weights = np.array([1., 0.5, 0.5, 0.25, 0.25, 0.125, 0.125])
            self.ds_loss_weights = self.ds_loss_weights / self.ds_loss_weights.sum()
            self.loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)

            self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] + f"_stage{self.stage}")

            if training:
                self.dl_tr, self.dl_val = self.get_basic_generators()
                if self.unpack_data:
                    unpack_dataset(self.folder_with_preprocessed_data)
                self.tr_gen, self.val_gen = get_moreDA_augmentation(
                    self.dl_tr,
                    self.dl_val,
                    self.data_aug_params['patch_size_for_spatialtransform'],
                    self.data_aug_params,
                    deep_supervision_scales=self.deep_supervision_scales,
                    pin_memory=self.pin_memory,
                    use_nondetMultiThreadedAugmenter=False,
                )

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()
            assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        else:
            self.print_to_log_file('Already initialized; skipping re-init.')

        self.was_initialized = True
        self._global_step = 0
        self._epoch_loss_sum = 0.0
        self._epoch_loss_cnt = 0
        self._ema_loss = None

        if getattr(self, "wandb_enabled", False) and wandb is not None:
            try:
                api_key = os.environ.get("WANDB_API_KEY")
                if api_key:
                    try:
                        wandb.login(key=api_key, relogin=False)
                    except Exception:
                        pass
                proj = self.wandb_project or os.environ.get("WANDB_PROJECT", "CSE6250_MNet_Reproduction")
                name = self.wandb_run_name or f"{self.__class__.__name__}_task{getattr(self, 'task', 'NA')}_fold{self.fold}"
                group = self.wandb_group or os.environ.get("WANDB_GROUP")
                cfg = {
                    "trainer": self.__class__.__name__,
                    "task": getattr(self, "task", None),
                    "fold": self.fold,
                    "plans": self.plans.get('plans_name', 'unknown') if isinstance(self.plans, dict) else "unknown",
                    "batch_dice": self.batch_dice,
                    "initial_epoch": self.epoch,
                    "plans_file": self.plans_file,
                    "output_folder": self.output_folder,
                    "dataset_directory": self.dataset_directory,
                }
                init_kwargs = {"project": proj, "name": name, "reinit": True, "config": cfg}
                if group is not None:
                    init_kwargs["group"] = group
                self._wb_run = wandb.init(**init_kwargs)

                try:
                    wandb.define_metric("global_step")
                    wandb.define_metric("epoch")
                    wandb.define_metric("loss/iter", step_metric="global_step")
                    wandb.define_metric("loss/iter_ema", step_metric="global_step")
                    wandb.define_metric("loss/epoch_mean", step_metric="epoch")
                    wandb.define_metric("loss/train", step_metric="epoch")
                    wandb.define_metric("loss/val", step_metric="epoch")
                    wandb.define_metric("dice/*", step_metric="epoch")
                    wandb.define_metric("lr", step_metric="epoch")
                except Exception:
                    pass

                try:
                    self._wb_run.watch(self.network, log_freq=100)
                except Exception:
                    pass

                self._wandb_log_architecture()
            except Exception as exc:
                self._wb_run = None
                self.print_to_log_file(f"W&B init skipped: {exc}")

    def initialize_network(self):
        net = MNetVMamba(
            self.num_input_channels,
            self.num_classes,
            kn=(32, 48, 64, 80, 96),
            ds=True,
            FMU_mode='sub',
            vmamba_cfg=self._vmamba_cfg(),
        )
        if torch.cuda.is_available():
            net.cuda()
        net.inference_apply_nonlin = softmax_helper

        total_params = sum(p.numel() for p in net.parameters())
        trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        self.print_to_log_file(f"Model params (VMamba): total={total_params:,}, trainable={trainable_params:,}")

        self.network = net

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None
        self.optimizer = torch.optim.SGD(
            self.network.parameters(),
            self.initial_lr,
            weight_decay=self.weight_decay,
            momentum=0.99,
            nesterov=True,
        )
        self.lr_scheduler = None

    def run_online_evaluation(self, output, target):
        if isinstance(target, (list, tuple)):
            target = target[0]
        if isinstance(output, (list, tuple)):
            output = output[0]
        return super().run_online_evaluation(output, target)

    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True,
                 step_size: float = 0.5, save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True):
        """Disable deep supervision during validation for proper inference."""
        ds = self.network.do_ds
        self.network.do_ds = False
        ret = super().validate(do_mirroring, use_sliding_window, step_size, save_softmax, use_gaussian,
                               overwrite, validation_folder_name, debug, all_in_gpu,
                               segmentation_export_kwargs, run_postprocessing_on_folds)
        self.network.do_ds = ds
        return ret

    def predict_preprocessed_data_return_seg_and_softmax(self, data: np.ndarray, do_mirroring: bool = True,
                                                         mirror_axes: Tuple[int] = None, use_sliding_window: bool = True,
                                                         step_size: float = 0.5, use_gaussian: bool = True,
                                                         pad_border_mode: str = 'constant', pad_kwargs: dict = None,
                                                         all_in_gpu: bool = False, verbose: bool = True,
                                                         mixed_precision=True) -> Tuple[np.ndarray, np.ndarray]:
        """Disable deep supervision during prediction for proper inference."""
        ds = self.network.do_ds
        self.network.do_ds = False
        ret = super().predict_preprocessed_data_return_seg_and_softmax(
            data, do_mirroring, mirror_axes, use_sliding_window, step_size, use_gaussian,
            pad_border_mode, pad_kwargs, all_in_gpu, verbose, mixed_precision
        )
        self.network.do_ds = ds
        return ret

    def setup_DA_params(self):
        self.deep_supervision_scales = [[1, 1, 1]] + list(
            list(i) for i in 1 / np.cumprod(np.vstack(self.net_num_pool_op_kernel_sizes), axis=0)
        )[:-1]

        if self.threeD:
            self.data_aug_params = default_3D_augmentation_params
            self.data_aug_params['rotation_x'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_y'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_z'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
        else:
            self.do_dummy_2D_aug = False
            if max(self.patch_size) / min(self.patch_size) > 1.5:
                default_2D_augmentation_params['rotation_x'] = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
            self.data_aug_params = default_2D_augmentation_params

        self.data_aug_params["mask_was_used_for_normalization"] = self.use_mask_for_norm
        if self.do_dummy_2D_aug:
            self.basic_generator_patch_size = get_patch_size(
                self.patch_size[1:],
                self.data_aug_params['rotation_x'],
                self.data_aug_params['rotation_y'],
                self.data_aug_params['rotation_z'],
                self.data_aug_params['scale_range'],
            )
            self.basic_generator_patch_size = np.array([self.patch_size[0]] + list(self.basic_generator_patch_size))
        else:
            self.basic_generator_patch_size = get_patch_size(
                self.patch_size,
                self.data_aug_params['rotation_x'],
                self.data_aug_params['rotation_y'],
                self.data_aug_params['rotation_z'],
                self.data_aug_params['scale_range'],
            )

        self.data_aug_params["scale_range"] = (0.7, 1.4)
        self.data_aug_params["do_elastic"] = False
        self.data_aug_params['selected_seg_channels'] = [0]
        self.data_aug_params['patch_size_for_spatialtransform'] = self.patch_size
        self.data_aug_params["num_cached_per_thread"] = 2

    # ------------------------------------------------------------------
    # Training / validation loops follow base implementation
    # ------------------------------------------------------------------
    # The rest of the methods (`run_iteration`, `validate`, etc.) mirror the parent class
    # with only minimal additions (logging). We reuse the original implementation.

    def run_iteration(self, data_generator, do_backprop: bool = True, run_online_evaluation: bool = False):
        data_dict = next(data_generator)
        data = maybe_to_torch(data_dict['data'])
        target = maybe_to_torch(data_dict['target'])

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

        self.optimizer.zero_grad()

        if self.fp16:
            if self.amp_grad_scaler is None:
                from torch.cuda.amp import GradScaler
                self.amp_grad_scaler = GradScaler()
            with autocast():
                output = self.network(data)
                if not isinstance(output, (list, tuple)):
                    output = (output,)
                loss = self.loss(output, target)
            if do_backprop:
                assert hasattr(self, "amp_grad_scaler"), "GradScaler expected when fp16=True"
                self.amp_grad_scaler.scale(loss).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            output = self.network(data)
            if not isinstance(output, (list, tuple)):
                output = (output,)
            loss = self.loss(output, target)
            if do_backprop:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        loss_val = float(loss.detach().cpu().numpy())
        self._epoch_loss_sum += loss_val
        self._epoch_loss_cnt += 1
        beta = 0.98
        self._ema_loss = loss_val if self._ema_loss is None else (beta * self._ema_loss + (1 - beta) * loss_val)

        self._global_step += 1
        if wandb is not None and self._wb_run is not None:
            try:
                log_dict = {"global_step": self._global_step, "loss/iter": loss_val}
                if self._ema_loss is not None:
                    log_dict["loss/iter_ema"] = float(self._ema_loss)
                wandb.log(log_dict, step=self._global_step)
            except Exception:
                pass

        del data, target
        return loss_val

    def maybe_update_lr(self, epoch=None):
        """Update learning rate with polynomial schedule."""
        ep = self.epoch + 1 if epoch is None else epoch
        self.optimizer.param_groups[0]['lr'] = poly_lr(ep, self.max_num_epochs, self.initial_lr, 0.9)
        self.print_to_log_file("lr:", np.round(self.optimizer.param_groups[0]['lr'], decimals=6))

    def run_training(self):
        """Run training and ensure clean exit."""
        try:
            ret = super().run_training()
            return ret
        finally:
            # Cleanup WandB to allow instance to idle
            if wandb is not None and self._wb_run is not None:
                try:
                    self._wb_run.finish()
                    self._wb_run = None
                except Exception:
                    pass

    def on_epoch_end(self):
        """Called at the end of each epoch - logs metrics to WandB."""
        super().on_epoch_end()
        
        # epoch mean loss
        epoch_mean = (self._epoch_loss_sum / max(1, self._epoch_loss_cnt))
        self._epoch_loss_sum = 0.0
        self._epoch_loss_cnt = 0

        logs = {
            "epoch": int(self.epoch),
            "lr": float(self.optimizer.param_groups[0]["lr"]),
            "loss/epoch_mean": float(epoch_mean),
        }

        # Log train and validation losses
        try:
            if hasattr(self, "all_tr_losses") and len(self.all_tr_losses) > 0:
                logs["loss/train"] = float(self.all_tr_losses[-1])
            if hasattr(self, "all_val_losses") and len(self.all_val_losses) > 0:
                logs["loss/val"] = float(self.all_val_losses[-1])
        except Exception:
            pass

        # Log Dice scores from online eval (mean + per-class if cached)
        try:
            if hasattr(self, "all_val_eval_metrics") and len(self.all_val_eval_metrics) > 0:
                last = self.all_val_eval_metrics[-1]
                mean_dc = float(getattr(self, "_last_online_eval_dc_mean",
                                        last if isinstance(last, (int, float)) else 0.0))
                logs["dice/mean"] = mean_dc
                if hasattr(self, "_last_online_eval_dc_per_class"):
                    for i, d in enumerate(self._last_online_eval_dc_per_class):
                        logs[f"dice/class_{i}"] = float(d)
        except Exception:
            pass

        # Log to WandB (don't pass step - let define_metric handle it via step_metric="epoch")
        if wandb is not None and self._wb_run is not None:
            try:
                wandb.log(logs)
            except Exception as e:
                self.print_to_log_file(f"WandB logging failed: {e}")

        # Continue training check
        continue_training = self.epoch < self.max_num_epochs
        if self.epoch == 100 and hasattr(self, "all_val_eval_metrics") and len(self.all_val_eval_metrics) > 0:
            if self.all_val_eval_metrics[-1] == 0:
                self.optimizer.param_groups[0]["momentum"] = 0.95
                self.network.apply(InitWeights_He(1e-2))
                self.print_to_log_file("Reduced momentum to 0.95 and reinitialized weights at epoch 100 due to 0 Dice.")
        return continue_training

    # Remaining lifecycle methods mirror parent implementation; re-use defaults where possible.


if __name__ == "__main__":  # pragma: no cover
    # Quick smoke test
    trainer = myTrainer_reproduction_WandB_2("plans.pkl", 0)
    trainer.vmamba_enabled = True
    trainer.vm_down_stages = [2, 3]
    trainer.vm_up_stages = [2]
    trainer.vm_bottleneck_stages = [4]
    trainer.initialize_network()
    x = torch.randn(1, trainer.num_input_channels, 16, 96, 96)
    with torch.no_grad():
        out = trainer.network(x)
    if isinstance(out, (list, tuple)):
        print(out[0].shape)
    else:
        print(out.shape)

