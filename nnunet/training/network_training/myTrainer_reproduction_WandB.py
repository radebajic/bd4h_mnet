import numpy as np
import torch
from typing import Tuple
from collections import OrderedDict
import torch.backends.cudnn as cudnn
from omegaconf import DictConfig

import numpy as np

from nnunet.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.training.data_augmentation.default_data_augmentation import (
    default_2D_augmentation_params,
    get_patch_size,
    default_3D_augmentation_params
)
from nnunet.training.dataloading.dataset_loading import unpack_dataset
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.utilities.nd_softmax import softmax_helper
from sklearn.model_selection import KFold
from torch import nn
from torch.cuda.amp import autocast
# from torch.amp import GradScaler, autocast
from nnunet.training.learning_rate.poly_lr import poly_lr
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.network_architecture.reproduction_mnet.mnet import MNet

from nnunet.network_architecture.reproduction_mnet.mnet import MNet


# WandB logging
import os
try:
    import wandb
except Exception:
    wandb = None

class myTrainer_reproduction_WandB(nnUNetTrainer):
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
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)

        # Trainer settings from explicit args
        self.max_num_epochs = max_num_epochs
        self.gated_fusion = gated_fusion

        self.initial_lr = 1e-2
        self.deep_supervision_scales = None
        self.ds_loss_weights = None
        self.pin_memory = True

        # Efficiency toggles
        self.width_mult = width_mult
        self.use_sep3d = use_sep3d
        self.use_checkpoint = use_checkpoint
        self.cat_reduce = cat_reduce

        # VMamba defaults (Hydra may override later)
        self.vm_down_stages = []
        self.vm_up_stages = []
        self.vm_bottleneck_stages = []
        self.axial_reduce = 0.5
        
        # W&B defaults (Hydra can override these attributes after construction)
        self.wandb_enabled: bool = False
        self.wandb_project: str | None = None
        self.wandb_run_name: str | None = None
        self._wb_run = None
        self._wb_run = None
        self._global_step = 0              # for per-iteration logging
        self._ema_loss = None
        self._ema_beta = 0.98
        self._epoch_loss_sum = 0.0
        self._epoch_loss_cnt = 0

        # Autotune cuDNN for mostly-fixed patch shapes
        cudnn.benchmark = True
        
        # Log the configuration
        self.print_to_log_file(f"Training config: max_epochs={self.max_num_epochs}, gated_fusion={self.gated_fusion}")
    
    def _wandb_log_architecture(self):
        if wandb is None or getattr(self, "_wb_run", None) is None or not hasattr(self, "network"):
            return
        # collect architecture knobs (from trainer attrs + network)
        arch_cfg = {
            "arch/network": self.network.__class__.__name__,
            "arch/width_mult": getattr(self, "width_mult", None),
            "arch/gated_fusion": getattr(self, "gated_fusion", None),
            "arch/use_sep3d": getattr(self, "use_sep3d", None),
            "arch/use_checkpoint": getattr(self, "use_checkpoint", None),
            "arch/cat_reduce": getattr(self, "cat_reduce", None),
            "arch/vmamba/down_stages": list(getattr(self, "vm_down_stages", [])),
            "arch/vmamba/up_stages": list(getattr(self, "vm_up_stages", [])),
            "arch/vmamba/bottleneck_stages": list(getattr(self, "vm_bottleneck_stages", [])),
            "arch/vmamba/axial_reduce": getattr(self, "axial_reduce", None),
        }
        # parameter counts
        try:
            total_params = sum(p.numel() for p in self.network.parameters())
            trainable_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
            arch_cfg["arch/params_total_m"] = round(total_params / 1e6, 3)
            arch_cfg["arch/params_trainable_m"] = round(trainable_params / 1e6, 3)
        except Exception:
            pass

        # push to W&B config for filtering
        try:
            wandb.config.update(arch_cfg, allow_val_change=True)
        except Exception:
            pass

        # also log once as metrics for easy charting
        try:
            wandb.log({k: v for k, v in arch_cfg.items() if isinstance(v, (int, float, str))}, step=getattr(self, "epoch", 0))
        except Exception:
            pass

        # upload a readable architecture text file
        try:
            arch_dir = Path(self.output_folder)
            arch_dir.mkdir(parents=True, exist_ok=True)
            arch_txt = arch_dir / "architecture.txt"
            with open(arch_txt, "w") as f:
                f.write(str(self.network))
            art = wandb.Artifact("architecture", type="text")
            art.add_file(str(arch_txt), name="architecture.txt")
            self._wb_run.log_artifact(art)
        except Exception:
            pass

    def initialize(self, training=True, force_load_plans=False):
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None):
                self.load_plans_file()

            self.process_plans(self.plans)
            self.setup_DA_params()

            # DS scales tuned for anisotropic z
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

            self.folder_with_preprocessed_data = join(
                self.dataset_directory, self.plans['data_identifier'] + f"_stage{self.stage}"
            )

            if training:
                self.dl_tr, self.dl_val = self.get_basic_generators()
                if self.unpack_data:
                    print("unpacking dataset")
                    unpack_dataset(self.folder_with_preprocessed_data)
                    print("done")
                self.tr_gen, self.val_gen = get_moreDA_augmentation(
                    self.dl_tr, self.dl_val,
                    self.data_aug_params['patch_size_for_spatialtransform'],
                    self.data_aug_params,
                    deep_supervision_scales=self.deep_supervision_scales,
                    pin_memory=self.pin_memory,
                    use_nondetMultiThreadedAugmenter=False
                )

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()
            assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        else:
            self.print_to_log_file('Already initialized; skipping re-init.')

        self.was_initialized = True
        
         # init counters for logging
        self._global_step = 0
        self._epoch_loss_sum = 0.0
        self._epoch_loss_cnt = 0
        self._ema_loss = None 
        
        # --- W&B init (minimal) ---
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
                self._wb_run = wandb.init(project=proj, name=name, reinit=True, config=cfg)

                # define per-metric step mapping to avoid step conflicts
                try:
                    wandb.define_metric("global_step")
                    wandb.define_metric("epoch")

                    wandb.define_metric("loss/iter", step_metric="global_step")
                    wandb.define_metric("loss/iter_ema", step_metric="global_step")

                    wandb.define_metric("loss/epoch_mean", step_metric="epoch")
                    wandb.define_metric("loss/train", step_metric="epoch")
                    wandb.define_metric("loss/val", step_metric="epoch")
                    wandb.define_metric("dice/*", step_metric="epoch")
                except Exception:
                    pass

                try:
                    self._wb_run.watch(self.network, log_freq=100)
                except Exception:
                    pass
                
                # NEW: log architecture once
                self._wandb_log_architecture()
                
            except Exception as e:
                self._wb_run = None
                self.print_to_log_file(f"W&B init skipped: {e}")


    def initialize_network(self):
        self.network = MNet(
            self.num_input_channels, self.num_classes,
            kn=(32, 48, 64, 80, 96),
            ds=True, FMU='sub',
            width_mult=getattr(self, "width_mult", 1.0),
            use_sep3d=getattr(self, "use_sep3d", False),
            use_checkpoint=getattr(self, "use_checkpoint", False),
            cat_reduce=getattr(self, "cat_reduce", False),
            gated_fusion=getattr(self, "gated_fusion", None),
            # NEW: pass per-stage switches
            vm_down_stages=getattr(self, "vm_down_stages", []),
            vm_up_stages=getattr(self, "vm_up_stages", []),
            vm_bottleneck_stages=getattr(self, "vm_bottleneck_stages", []),
            axial_reduce=getattr(self, "axial_reduce", 0.5),
        )
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

        total_params = sum(p.numel() for p in self.network.parameters())
        trainable_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        self.print_to_log_file(f"Model params: total={total_params:,}, trainable={trainable_params:,}")


        # # Optional PyTorch 2 compile (safe guard)
        # try:
        #     if hasattr(torch, "compile"):
        #         self.network = torch.compile(self.network, mode="max-autotune")
        # except Exception as e:
        #     self.print_to_log_file(f"torch.compile not enabled: {e}")

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None
        self.optimizer = torch.optim.SGD(
            self.network.parameters(), self.initial_lr,
            weight_decay=self.weight_decay,
            momentum=0.99, nesterov=True
        )
        self.lr_scheduler = None

    def run_online_evaluation(self, output, target):
        target = target[0]
        output = output[0]
        return super().run_online_evaluation(output, target)

    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True,
                 step_size: float = 0.5, save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True):
        ds = self.network.do_ds
        self.network.do_ds = False
        ret = super().validate(do_mirroring, use_sliding_window, step_size, save_softmax, use_gaussian,
                               overwrite, validation_folder_name, debug, all_in_gpu,
                               segmentation_export_kwargs, run_postprocessing_on_folds)
        self.network.do_ds = ds
        # log validation dice immediately if available
        try:
            if wandb is not None and self._wb_run is not None:
                logs = {}
                last = self.all_val_eval_metrics[-1] if getattr(self, "all_val_eval_metrics", None) else ret
                if isinstance(last, dict):
                    if 'mean' in last:
                        logs["dice/mean"] = float(last['mean'])
                    if 'all_dices' in last and last['all_dices'] is not None:
                        for i, d in enumerate(last['all_dices']):
                            logs[f"dice/class_{i}"] = float(d)
                elif isinstance(last, (list, tuple)) and len(last) > 0:
                    # common tuple: (mean, per_class, ...) or list
                    try:
                        logs["dice/mean"] = float(last[0])
                    except Exception:
                        pass
                    if len(last) > 1 and isinstance(last[1], (list, tuple, np.ndarray)):
                        for i, d in enumerate(last[1]):
                            logs[f"dice/class_{i}"] = float(d)
                if logs:
                    wandb.log(logs, step=self.epoch)
        except Exception:
            pass
        return ret

    def predict_preprocessed_data_return_seg_and_softmax(self, data: np.ndarray, do_mirroring: bool = True,
                                                         mirror_axes: Tuple[int] = None, use_sliding_window: bool = True,
                                                         step_size: float = 0.5, use_gaussian: bool = True,
                                                         pad_border_mode: str = 'constant', pad_kwargs: dict = None,
                                                         all_in_gpu: bool = False, verbose: bool = True,
                                                         mixed_precision=True) -> Tuple[np.ndarray, np.ndarray]:
        ds = self.network.do_ds
        self.network.do_ds = False
        ret = super().predict_preprocessed_data_return_seg_and_softmax(
            data, do_mirroring, mirror_axes, use_sliding_window, step_size, use_gaussian,
            pad_border_mode, pad_kwargs, all_in_gpu, verbose, mixed_precision
        )
        self.network.do_ds = ds
        return ret

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        data_dict = next(data_generator)
        data = maybe_to_torch(data_dict['data'])
        target = maybe_to_torch(data_dict['target'])

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

        self.optimizer.zero_grad()

        if self.fp16:
            from torch.cuda.amp import GradScaler, autocast
            if not hasattr(self, "amp_grad_scaler"):
                self.amp_grad_scaler = GradScaler()
            with autocast():
                output = self.network(data)
                if not isinstance(output, (list, tuple)):
                    output = (output,)
                del data
                loss = self.loss(output, target)
            if do_backprop:
                self.amp_grad_scaler.scale(loss).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            output = self.network(data)
            if not isinstance(output, (list, tuple)):
                output = (output,)
            del data
            loss = self.loss(output, target)
            if do_backprop:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)
            
        loss_val = float(loss.detach().cpu().numpy())

        # update running epoch stats
        self._epoch_loss_sum += loss_val
        self._epoch_loss_cnt += 1
        
        # update EMA safely
        beta = 0.98
        self._ema_loss = loss_val if self._ema_loss is None else (beta * self._ema_loss + (1 - beta) * loss_val)

        # per-iteration logging with a global step
        self._global_step += 1
        try:
            if wandb is not None and self._wb_run is not None:
                log_dict = {
                    "global_step": self._global_step,
                    "loss/iter": loss_val,
                }
                # only include EMA if available
                if self._ema_loss is not None:
                    log_dict["loss/iter_ema"] = float(self._ema_loss)

                # since we defined step_metric for loss/iter, keeping step for consistent _step timeline
                wandb.log(log_dict, step=self._global_step)
        except Exception:
            pass

        del target
        return loss.detach().cpu().numpy()

    def setup_DA_params(self):
        self.deep_supervision_scales = [[1, 1, 1]] + list(
            list(i) for i in 1 / np.cumprod(np.vstack(self.net_num_pool_op_kernel_sizes), axis=0)
        )[:-1]

        if self.threeD:
            self.data_aug_params = default_3D_augmentation_params
            self.data_aug_params['rotation_x'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_y'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_z'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            if self.do_dummy_2D_aug:
                self.data_aug_params["dummy_2D"] = True
                self.data_aug_params["elastic_deform_alpha"] = default_2D_augmentation_params["elastic_deform_alpha"]
                self.data_aug_params["elastic_deform_sigma"] = default_2D_augmentation_params["elastic_deform_sigma"]
                self.data_aug_params["rotation_x"] = default_2D_augmentation_params["rotation_x"]
        else:
            self.do_dummy_2D_aug = False
            if max(self.patch_size) / min(self.patch_size) > 1.5:
                default_2D_augmentation_params['rotation_x'] = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
            self.data_aug_params = default_2D_augmentation_params

        self.data_aug_params["mask_was_used_for_normalization"] = self.use_mask_for_norm

        if self.do_dummy_2D_aug:
            self.basic_generator_patch_size = get_patch_size(
                self.patch_size[1:], self.data_aug_params['rotation_x'], self.data_aug_params['rotation_y'],
                self.data_aug_params['rotation_z'], self.data_aug_params['scale_range']
            )
            self.basic_generator_patch_size = np.array([self.patch_size[0]] + list(self.basic_generator_patch_size))
        else:
            self.basic_generator_patch_size = get_patch_size(
                self.patch_size, self.data_aug_params['rotation_x'], self.data_aug_params['rotation_y'],
                self.data_aug_params['rotation_z'], self.data_aug_params['scale_range']
            )

        self.data_aug_params["scale_range"] = (0.7, 1.4)
        self.data_aug_params["do_elastic"] = False
        self.data_aug_params['selected_seg_channels'] = [0]
        self.data_aug_params['patch_size_for_spatialtransform'] = self.patch_size
        self.data_aug_params["num_cached_per_thread"] = 2

    def maybe_update_lr(self, epoch=None):
        ep = self.epoch + 1 if epoch is None else epoch
        self.optimizer.param_groups[0]['lr'] = poly_lr(ep, self.max_num_epochs, self.initial_lr, 0.9)
        self.print_to_log_file("lr:", np.round(self.optimizer.param_groups[0]['lr'], decimals=6))

    def on_epoch_end(self):
        super().on_epoch_end()
        
        # epoch mean loss
        epoch_mean = (self._epoch_loss_sum / max(1, self._epoch_loss_cnt))
        self._epoch_loss_sum = 0.0
        self._epoch_loss_cnt = 0

        logs = {
            "epoch": int(self.epoch),  # let W&B use this as the step for epoch metrics
            "lr": float(self.optimizer.param_groups[0]["lr"]),
            "loss/epoch_mean": float(epoch_mean),
        }

        # prefer nnU-Net’s tracked losses
        try:
            if hasattr(self, "all_tr_losses") and len(self.all_tr_losses) > 0:
                logs["loss/train"] = float(self.all_tr_losses[-1])
            if hasattr(self, "all_val_losses") and len(self.all_val_losses) > 0:
                logs["loss/val"] = float(self.all_val_losses[-1])
        except Exception:
            pass

        # Dice from online eval (mean + per-class if cached)
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

        # single epoch log (no conflicting step)
        try:
            if wandb is not None and self._wb_run is not None:
                wandb.log(logs)  # do not pass `step`; W&B uses define_metric step_metric
        except Exception:
            pass

        continue_training = self.epoch < self.max_num_epochs

        if not continue_training:
            try:
                if self._wb_run is not None:
                    self._wb_run.finish()
            except Exception:
                pass
            self._wb_run = None

        if self.epoch == 100 and self.all_val_eval_metrics[-1] == 0:
            self.optimizer.param_groups[0]["momentum"] = 0.95
            self.network.apply(InitWeights_He(1e-2))
            self.print_to_log_file("Reduced momentum to 0.95 and reinitialized weights at epoch 100 due to 0 Dice.")
        return continue_training
    
    # finalize (idempotent)
    def on_train_end(self):
        super().on_train_end()
        run = getattr(self, "_wb_run", None)
        self._wb_run = None           # prevent double-finish (atexit/signal)
        try:
            if run is not None:
                run.finish()
        except Exception as e:
            self.print_to_log_file(f"W&B finish failed: {e}")

# --- IGNORE ---