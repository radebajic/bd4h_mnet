import hydra
from omegaconf import DictConfig
from nnunet.training.network_training.myTrainer_reproduction_wandb import myTrainer_reproduction
import os
from pathlib import Path

# Project-local nnU-Net paths (adjust if you want other locations)
_proj_root = Path(__file__).resolve().parent
os.environ.setdefault("nnUNet_raw_data_base",
                      os.path.join(_proj_root, "nnUNet_raw"))
os.environ.setdefault("nnUNet_preprocessed", os.path.join(
    _proj_root, "nnUNet_preprocessed"))
os.environ.setdefault("RESULTS_FOLDER", os.path.join(
    _proj_root, "nnUNet_results"))


@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg: DictConfig) -> None:
    """Training script with Hydra configuration."""

    # Create trainer with Hydra config
    trainer = myTrainer_reproduction(
            hydra_cfg=cfg
    )

    # Initialize and run training
    trainer.initialize()
    trainer.run_training()


if __name__ == "__main__":
    train()
