#!/bin/bash
#SBATCH --gpus=h100-47:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=00:59:00
#SBATCH --output=outputs/t5_generate-%j.out
#SBATCH --error=outputs/t5_generate-%j.err
#SBATCH --job-name=debug
#SBATCH --mail-user=<username>@comp.nus.edu.sg
#SBATCH --mail-type=END,FAIL

srun /bin/bash -c \
"
cd \"\${SLURM_SUBMIT_DIR:-$PWD}\"
export WANDB_PROJECT=text-mining-baselines
uv run python method/infer_model.py --config method/configs/simple.yaml
"
