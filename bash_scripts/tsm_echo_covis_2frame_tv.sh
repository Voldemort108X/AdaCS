#!/bin/bash
#SBATCH --job-name=cvpr2024_covis
#SBATCH --out="./slurm_out/slurm-%j.out"
#SBATCH --time=08:00:00
#SBATCH --gpus=a5000:1
#SBATCH --mem=64G
#SBATCH --partition gpu

pwd
nvidia-smi
module load miniconda
source activate torch_flow
python transmorph_torch/train_2frame_covis_tv.py --dataset 'Echo'  --bidir --model-dir '../../Models/cvpr2024_covis/Echo/tsm_echo_2frame_covis_alpha_0.01_lmbd_0.025_wse_20_tv_adabeta_0.1' --motion-loss-type 'wmse' --covis-loss-type 'coviswmse' --batch-size 8 --accumulation_steps 2 --epochs 150 --lambda 0.025 --warm_start --wandb-name 'tsm_echo_2frame_covis_alpha_0.01_lmbd_0.025_wse_10_tv_adabeta_0.1' --alpha 0.01 --warm_start_epoch 10 --beta 0.1
