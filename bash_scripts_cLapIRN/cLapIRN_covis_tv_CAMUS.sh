#!/bin/bash
#SBATCH --job-name=cvpr2024_covis_tv
#SBATCH --out="./slurm_out/slurm-%j.out"
#SBATCH --time=04:00:00
#SBATCH --gpus=a5000:1
#SBATCH --mem=64G
#SBATCH --partition gpu

pwd
nvidia-smi
module load miniconda
source activate torch_flow
python Conditional_LapIRN/train_2D_mse_covis_tv.py --dataset 'CAMUS' --model-dir '../../Models/cvpr2024_covis/CAMUS/cLapIRN_covis_tv_alpha_0.01_beta_0.1'  --wandb-name 'cLapIRN_CAMUS_covis_tv_alpha_0.01_beta_0.1' --beta 0.1 --alpha 0.01 
