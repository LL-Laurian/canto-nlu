#!/bin/bash
#SBATCH --account=rrg-annielee
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%A_%a.out
#############################################################
# One fine-tuning run per Slurm array task, submitted by slurm/hpo.py: array index i runs line i+1
# of RUNS_FILE, which holds "MODEL_DIR TASK [run.py args]" for slurm/finetune.sh.
#
#   sbatch --array=0-9 --job-name=hpo-pos slurm/hpo_array.sh logs/hpo_runs/pos-20261102-120000.txt
#############################################################
RUNS_FILE=$1
read -r -a RUN <<< "$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$RUNS_FILE")"
bash slurm/finetune.sh "${RUN[@]}"
