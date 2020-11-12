#!/bin/bash
#SBATCH --job-name=3Adj-EGNNA
#SBATCH --partition=prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --array=0-1
#SBATCH --output=slurm_output/color_rgb_lstsrtmssm_egnna_top10_excel%A_%a.out

source activate py_env2
module load cuda/10.0

export PYTHONPATH="${PYTHONPATH}:/homes/svincenzi/.conda/envs/py_env2/bin/python"

srun python -u main.py --id_optim=${SLURM_ARRAY_TASK_ID}