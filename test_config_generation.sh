#!/bin/bash
# Quick test to verify config generation works without running full sweep

echo "Testing config generation..."

cd /teamspace/studios/this_studio/bd4h_mnet-1

python3 << 'PYTHON_EOF'
from pathlib import Path
import yaml

models = {
    "baseline": {
        "gated_fusion": None, 
        "bottleneck_stages": [],
        "trainer_name": "myTrainer_reproduction_WandB_baseline"
    },
    "sg": {
        "gated_fusion": "spatial", 
        "bottleneck_stages": [],
        "trainer_name": "myTrainer_reproduction_WandB_sg"
    },
    "vmamba": {
        "gated_fusion": None, 
        "bottleneck_stages": [3, 4, 5],
        "trainer_name": "myTrainer_reproduction_WandB_vmamba"
    },
    "sg_vmamba": {
        "gated_fusion": "spatial", 
        "bottleneck_stages": [3, 4, 5],
        "trainer_name": "myTrainer_reproduction_WandB_sg_vmamba"
    },
}

spacings = ["z1p0", "z2p2", "z4p0"]
folds = [0]
base_dir = Path("conf/training")

count = 0
for model_key, model_config in models.items():
    for z in spacings:
        config_dir = base_dir / f"lits_sweep_{model_key}" / z
        config_dir.mkdir(parents=True, exist_ok=True)
        
        for fold in folds:
            train_config = {
                "hydra": {
                    "job": {
                        "chdir": False,
                        "env_set": {
                            "WANDB_ENTITY": "xplstm",
                            "WANDB_PROJECT": "CSE6250_MNet_LiTS_Sweep",
                            "WANDB_GROUP": f"lits_{model_key}_{z}"
                        }
                    }
                },
                "trainer": {
                    "model": "3d_fullres",
                    "trainer_name": model_config["trainer_name"],
                    "task": "Task029_LITS",
                    "fold": fold,
                    "plans_identifier": f"nnUNetData_plans_v2.1_trgSp_{z}_yx0p9121",
                    "save_npz": True,
                    "validation_only": False,
                    "valbest": False,
                    "dataset_directory": "/teamspace/studios/this_studio/nnUNet_preprocessed",
                    "results_base": "/teamspace/studios/this_studio/nnUNet_results",
                    "fp16": True,
                    "deterministic": False,
                    "unpack_data": True,
                    "max_epochs": 150,
                    "gated_fusion": model_config["gated_fusion"],
                    "width_mult": 1.0,
                    "use_sep3d": False,
                    "use_checkpoint": False,
                    "cat_reduce": False,
                    "batch_size": 4,
                    "num_threads": 16,
                    "vmamba": {
                        "down_stages": [],
                        "up_stages": [],
                        "bottleneck_stages": model_config["bottleneck_stages"],
                        "axial_reduce": 0.5
                    },
                    "wandb": {
                        "enabled": True,
                        "project": "CSE6250_MNet_LiTS_Sweep",
                        "run_name": f"lits_{model_key}_{z}_fold{fold}",
                        "group": f"lits_{model_key}_{z}"
                    }
                }
            }
            
            with open(config_dir / f"best_arch_3_fold{fold}.yaml", 'w') as f:
                yaml.dump(train_config, f, default_flow_style=False, sort_keys=False)
            
            val_config = train_config.copy()
            val_config["trainer"]["validation_only"] = True
            val_config["trainer"]["valbest"] = True
            
            with open(config_dir / f"validate_best_arch3_fold{fold}.yaml", 'w') as f:
                yaml.dump(val_config, f, default_flow_style=False, sort_keys=False)
            
            count += 2  # train + validate
        
        print(f"✓ Created configs for: {model_key}/{z}")

print(f"\n✓ Successfully created {count} config files!")
print("\nConfig structure:")
for model_key in models.keys():
    print(f"  conf/training/lits_sweep_{model_key}/")
    for z in spacings:
        print(f"    {z}/")
        print(f"      best_arch_3_fold0.yaml")
        print(f"      validate_best_arch3_fold0.yaml")

PYTHON_EOF

echo ""
echo "✓ Config generation test passed!"
echo ""
echo "To run the full sweep, execute:"
echo "  ./run_full_sweep_overnight.sh"


