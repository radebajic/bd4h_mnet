#!/bin/bash
# Wrapper script to run training and ensure clean shutdown for auto-idle

CONFIG_NAME="${1:-best_arch_3_fixed}"
MULTIRUN="${2:-}"

echo "=== Starting Training ==="
echo "Config: $CONFIG_NAME"
echo "Time: $(date)"
echo ""

cd /teamspace/studios/this_studio/bd4h_mnet-1

# Run training
if [ "$MULTIRUN" == "-m" ]; then
    echo "Running multirun sweep..."
    python hydra_trainer.py --config-name "$CONFIG_NAME" -m
else
    echo "Running single config..."
    python hydra_trainer.py --config-name "$CONFIG_NAME"
fi

TRAIN_EXIT_CODE=$?

echo ""
echo "=== Training Finished ==="
echo "Exit code: $TRAIN_EXIT_CODE"
echo "Time: $(date)"

# Cleanup to allow auto-idle
echo ""
echo "=== Cleaning up for auto-idle ==="

# Give WandB time to sync final data
echo "Waiting for WandB sync..."
sleep 10

# Kill any lingering Python processes from this training
echo "Cleaning up processes..."
pkill -f "hydra_trainer.py" 2>/dev/null || true

# Clear GPU cache
echo "Clearing GPU cache..."
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# Wait a bit
sleep 5

# Final status
echo ""
echo "=== Final Status ==="
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null || echo "No GPU processes"

echo ""
if [ $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l) -eq 0 ]; then
    echo "✅ All clear! Instance will auto-idle in ~30 minutes"
    echo "   You can safely close Cursor now."
else
    echo "⚠️  Some GPU processes still running"
    echo "   Instance may not auto-idle immediately"
fi

echo ""
echo "=== Done ==="
exit $TRAIN_EXIT_CODE

