#!/bin/bash
#SBATCH --job-name=cvpr2024_dual_uncertainty
#SBATCH --out="./slurm_out/slurm-%j.out"
#SBATCH --time=08:00:00
#SBATCH --gpus=a5000:1
#SBATCH --mem=64G
#SBATCH --partition gpu

pwd
nvidia-smi
module load miniconda
source activate torch_flow
python diffusemorph_torch/train_2frame_adaframe.py  --dataset 'Echo' -p train -c diffusemorph_torch/config/diffuseMorph_train_3D.json  --motion-loss-type 'wmse' --batch-size 8 --epoch 150 --accumulation_steps 2 --model-dir '../../Models/cvpr2024_covis/Echo/dfm_echo_2frame_adaframe' --lambda_L 20 --gamma 1 --wandb-name 'dfm_echo_2frame_adaframe'