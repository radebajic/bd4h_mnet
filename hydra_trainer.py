import os
import hydra
from omegaconf import DictConfig
from nnunet.run.default_configuration import get_default_configuration
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.training.network_training.nnUNetTrainerCascadeFullRes import nnUNetTrainerCascadeFullRes
from nnunet.training.network_training.nnUNetTrainerV2_CascadeFullRes import nnUNetTrainerV2CascadeFullRes
from nnunet.run.load_pretrained_weights import load_pretrained_weights

from omegaconf import DictConfig, ListConfig
from collections.abc import Sequence

def _extract_best_objective(trainer) -> float:
    vals = []
    for m in getattr(trainer, "all_val_eval_metrics", []) or []:
        if isinstance(m, dict) and "mean" in m:
            vals.append(float(m["mean"]))
        elif isinstance(m, (list, tuple)) and len(m) > 0:
            try: vals.append(float(m[0]))
            except Exception: pass
        elif isinstance(m, (float, int)):
            vals.append(float(m))
    if vals: return max(vals)  # maximize Dice
    if getattr(trainer, "all_val_losses", None):
        try: return -float(min(trainer.all_val_losses))  # fallback: minimize val loss
        except Exception: pass
    return float("nan")

def _as_int_list(x):
    # Coerce YAML/CLI/Optuna inputs (including OmegaConf ListConfig) to list[int]
    if x is None:
        return []
    # Convert OmegaConf containers to native
    if isinstance(x, ListConfig):
        x = list(x)
    # Generic sequences (but not strings/bytes)
    if isinstance(x, Sequence) and not isinstance(x, (str, bytes)):
        return [int(i) for i in x]
    # Strings like "[]", "[2,3]" or "2,3"
    if isinstance(x, str):
        s = x.strip()
        if s == "" or s == "[]":
            return []
        s = s.strip("[]")
        return [int(t) for t in s.split(",") if t.strip() != ""]
    # Numbers
    if isinstance(x, (int, float)):
        return [int(x)]
    # Fallback: try to iterate
    try:
        return [int(i) for i in list(x)]
    except Exception:
        raise TypeError(f"Cannot coerce to list[int]: {type(x)} -> {x}")

@hydra.main(version_base=None, config_path="conf/training", config_name="quick")
def train(cfg: DictConfig) -> float:
    tc = cfg.get("trainer", {})

    # Mirrors: nnUNet_train <model> <trainer_name> <task> <fold> -p <plans_identifier>
    model = tc.get("model", "3d_fullres")
    trainer_name = tc.get("trainer_name", "myTrainer_reproduction_WandB")
    task = tc.get("task")
    fold = int(tc.get("fold", 0))
    plans_identifier = tc.get("plans_identifier")

    if not task or plans_identifier is None:
        raise ValueError("Set trainer.task and trainer.plans_identifier in quick.yaml")

    # Pass through paths like the CLI would
    if tc.get("dataset_directory"):
        os.environ["nnUNet_preprocessed"] = str(tc["dataset_directory"])
    if tc.get("results_base"):
        os.environ["RESULTS_FOLDER"] = str(tc["results_base"])

    # Resolve canonical nnU-Net config (keeps parity with nnUNet_train)
    plans_file, output_folder, dataset_directory, batch_dice, stage, trainer_class = get_default_configuration(
        model, task, trainer_name, plans_identifier
    )

    # Sanity checks (same as run_training.py)
    if model == "3d_cascade_fullres":
        assert issubclass(trainer_class, (nnUNetTrainerCascadeFullRes, nnUNetTrainerV2CascadeFullRes))
    else:
        assert issubclass(trainer_class, nnUNetTrainer)

    # Map flags like the CLI
    run_mixed_precision = bool(tc.get("fp16", True))
    deterministic = bool(tc.get("deterministic", False))
    decompress_data = bool(tc.get("unpack_data", True))  # True = unpack (default nnUNet behavior)

    # Instantiate trainer
    trainer = trainer_class(
        plans_file, fold,
        output_folder=output_folder,
        dataset_directory=dataset_directory,
        batch_dice=batch_dice,
        stage=stage,
        unpack_data=decompress_data,
        deterministic=deterministic,
        fp16=run_mixed_precision,
    )

    # Optional knobs for custom trainers (no-op for standard ones)
    if hasattr(trainer, "max_num_epochs"):
        trainer.max_num_epochs = int(tc.get("max_epochs", getattr(trainer, "max_num_epochs", 1)))

    # Normalize and set gated_fusion explicitly
    gf = tc.get("gated_fusion", None)
    if isinstance(gf, str) and gf.lower() in ("none", "null", "off", ""):
        gf = None
    if hasattr(trainer, "gated_fusion"):
        trainer.gated_fusion = gf  # type: ignore[attr-defined]

    # Set other simple knobs if provided
    for k in ("width_mult", "use_sep3d", "use_checkpoint", "cat_reduce"):
        if k in tc and hasattr(trainer, k):
            setattr(trainer, k, tc[k])

    # VMamba wiring: supports both preset flags and per-stage lists
    vm = tc.get("vmamba", {}) or {}

    # legacy/preset-style flags if your trainer/network supports them
    for name, attr in (("preset", "vmamba_preset"),
                       ("in_down", "vmamba_in_down"),
                       ("in_up", "vmamba_in_up"),
                       ("in_bottleneck", "vmamba_in_bottleneck")):
        if name in vm and hasattr(trainer, attr):
            setattr(trainer, attr, vm.get(name))

    # per-stage lists (set UNCONDITIONALLY so sweeps propagate)
    trainer.vm_down_stages = _as_int_list(vm.get("down_stages", []))
    trainer.vm_up_stages = _as_int_list(vm.get("up_stages", []))
    trainer.vm_bottleneck_stages = _as_int_list(vm.get("bottleneck_stages", []))

    def _validate_stage_range(name: str, values: Sequence[int], low: int, high: int) -> None:
        invalid = [v for v in values if v < low or v > high]
        if invalid:
            raise ValueError(
                f"trainer.vmamba.{name} must be between {low} and {high} inclusive; got invalid entries {invalid}"
            )

    _validate_stage_range("down_stages", trainer.vm_down_stages, 1, 4)
    _validate_stage_range("up_stages", trainer.vm_up_stages, 1, 4)
    _validate_stage_range("bottleneck_stages", trainer.vm_bottleneck_stages, 1, 5)
    if hasattr(trainer, "axial_reduce"):
        trainer.axial_reduce = float(vm.get("axial_reduce", getattr(trainer, "axial_reduce", 0.5)))  # type: ignore[attr-defined]

    # WandB toggles (consumed by your custom trainer)
    wb = tc.get("wandb", {})
    if hasattr(trainer, "wandb_enabled"):
        trainer.wandb_enabled = bool(wb.get("enabled", False))  # type: ignore[attr-defined]
        trainer.wandb_project = wb.get("project", None)  # type: ignore[attr-defined]
        trainer.wandb_run_name = wb.get("run_name", None)  # type: ignore[attr-defined]

    # --npz equivalent
    setattr(trainer, "save_npz", bool(tc.get("save_npz", False)))

    # Debug to confirm sweep values reach the trainer
    try:
      down = getattr(trainer, "vm_down_stages", None)
      up = getattr(trainer, "vm_up_stages", None)
      bottleneck = getattr(trainer, "vm_bottleneck_stages", None)
      print(f"[DEBUG] VMamba down={down} up={up} bn={bottleneck} "
          f"gf={getattr(trainer,'gated_fusion', None)} width_mult={getattr(trainer,'width_mult', None)}")
    except Exception:
        pass

    validation_only = bool(tc.get("validation_only", False))
    continue_training = bool(tc.get("continue_training", False))
    valbest = bool(tc.get("valbest", False))

    trainer.initialize(training=not validation_only)

    if validation_only:
        if valbest and hasattr(trainer, "load_best_checkpoint"):
            trainer.load_best_checkpoint(train=False)
        else:
            trainer.load_final_checkpoint(train=False)

        if not bool(tc.get("disable_validation_inference", False)):
            validation_folder = tc.get("validation_folder", "validation_raw")
            overwrite = bool(tc.get("val_disable_overwrite", True))
            run_postprocessing = not bool(tc.get("disable_postprocessing_on_folds", False))

            trainer.validate(
                save_softmax=bool(tc.get("save_npz", False)),
                validation_folder_name=validation_folder,
                overwrite=overwrite,
                run_postprocessing_on_folds=run_postprocessing,
            )

        return _extract_best_objective(trainer)

    if continue_training:
        trainer.load_latest_checkpoint()
    elif tc.get("pretrained_weights"):
        load_pretrained_weights(trainer.network, tc.get("pretrained_weights"))

    trainer.run_training()
    return _extract_best_objective(trainer)

if __name__ == "__main__":
    train()