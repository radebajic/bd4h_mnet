import os
import hydra
from omegaconf import DictConfig
from nnunet.run.default_configuration import get_default_configuration
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.training.network_training.nnUNetTrainerCascadeFullRes import nnUNetTrainerCascadeFullRes
from nnunet.training.network_training.nnUNetTrainerV2_CascadeFullRes import nnUNetTrainerV2CascadeFullRes

@hydra.main(version_base=None, config_path="conf/training", config_name="quick")
def train(cfg: DictConfig) -> None:
    tc = cfg.get("trainer", {})

    # Mirrors: nnUNet_train <model> <trainer_name> <task> <fold> -p <plans_identifier>
    model = tc.get("model", "3d_fullres")
    trainer_name = tc.get("trainer_name", "myTrainer_reproduction_WandB")
    task = tc.get("task")
    fold = tc.get("fold", 0)
    plans_identifier = tc.get("plans_identifier")

    if not task or plans_identifier is None:
        raise ValueError("Set trainer.task and trainer.plans_identifier in quick.yaml")

    # Make nnU-Net resolve paths exactly like CLI
    if tc.get("dataset_directory"):
        os.environ["nnUNet_preprocessed"] = str(tc["dataset_directory"])
    if tc.get("results_base"):
        os.environ["RESULTS_FOLDER"] = str(tc["results_base"])

    # Resolve plans_file, output folder, dataset_directory, batch_dice, stage, trainer_class
    plans_file, output_folder, dataset_directory, batch_dice, stage, trainer_class = get_default_configuration(
        model, task, trainer_name, plans_identifier
    )

    # Sanity (same asserts as run_training.py)
    if model == "3d_cascade_fullres":
        assert issubclass(trainer_class, (nnUNetTrainerCascadeFullRes, nnUNetTrainerV2CascadeFullRes))
    else:
        assert issubclass(trainer_class, nnUNetTrainer)

    # Map flags like the CLI
    run_mixed_precision = bool(tc.get("fp16", True))
    deterministic = bool(tc.get("deterministic", False))
    decompress_data = bool(tc.get("unpack_data", True))  # True = unpack (default nnUNet behavior)

    # Instantiate and then set optional custom attributes (keeps signature simple)
    trainer = trainer_class(
        plans_file, int(fold),
        output_folder=output_folder,
        dataset_directory=dataset_directory,
        batch_dice=batch_dice,
        stage=stage,
        unpack_data=decompress_data,
        deterministic=deterministic,
        fp16=run_mixed_precision,
    )

    # Optional knobs for your custom trainer (no-op for standard trainers)
    if hasattr(trainer, "max_num_epochs"):
        trainer.max_num_epochs = int(tc.get("max_epochs", getattr(trainer, "max_num_epochs", 1)))
    for k in ("gated_fusion", "width_mult", "use_sep3d", "use_checkpoint", "cat_reduce"):
        if hasattr(trainer, k) and k in tc:
            setattr(trainer, k, tc[k])

    # --npz equivalent
    setattr(trainer, "save_npz", bool(tc.get("save_npz", False)))
    
    wb = tc.get("wandb", {})
    if hasattr(trainer, "wandb_enabled"):
        trainer.wandb_enabled = bool(wb.get("enabled", False))
        trainer.wandb_project = wb.get("project", None)
        trainer.wandb_run_name = wb.get("run_name", None)
    
    trainer.initialize(training=True)
    trainer.run_training()

if __name__ == "__main__":
    train()