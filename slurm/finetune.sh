#!/bin/bash
#SBATCH --job-name=cantonlu-finetune
#SBATCH --account=rrg-annielee
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#############################################################
# One fine-tuning run through the unmodified run.py. Arguments after MODEL_DIR and TASK are
# passed to run.py as-is, so with none the run uses the original baseline settings.
#
#   sbatch slurm/finetune.sh models/cantomodernbert-mono sentiment                     # baseline
#   sbatch slurm/finetune.sh models/cantomodernbert-mono sentiment --lr=5e-5 --warmup_ratio=0.1 --weight_decay=0.01
#   sbatch --time=04:00:00 slurm/finetune.sh models/chinese-modernbert nli --epochs=3   # override resources per run
#
# Run from the repository root, after `mkdir -p logs`.
#############################################################
MODEL_DIR=$1
TASK=$2
shift 2

module load python/3.11 gcc arrow
source ~/envs/cantonlu-v5/bin/activate

# Models and data are on disk already: fail fast instead of reaching for the Hugging Face Hub
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME=/scratch/$USER/hf_cache

echo "job $SLURM_JOB_ID on $(hostname), commit $(git rev-parse --short HEAD)$(git diff --quiet || echo ' (uncommitted changes)')"
echo "model=$MODEL_DIR task=$TASK extra args: $*"
nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__)"

start=$(date +%s)
python run.py --task="$TASK" --model_dir="$MODEL_DIR" "$@"
echo "wall time: $(( $(date +%s) - start )) s"
# After the job ends, `seff $SLURM_JOB_ID` reports elapsed time and peak memory
