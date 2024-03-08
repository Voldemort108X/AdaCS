#!/bin/bash
#SBATCH --job-name=cvpr2024_covis
#SBATCH --out="./slurm_out/slurm-%j.out"
#SBATCH --time=04:00:00
#SBATCH --gpus=a5000:1
#SBATCH --mem=64G
#SBATCH --partition gpu

pwd
nvidia-smi
module load miniconda
source activate torch_flow
python transmorph_torch/train_2frame_covis_tv.py --dataset 'ACDC'  --bidir --model-dir '../../Models/cvpr2024_covis/ACDC/tsm_ACDC_2frame_covis_alpha_0.01_wse_20_tv_adabeta_0.75_init' --motion-loss-type 'wmse' --covis-loss-type 'coviswmse' --batch-size 8 --accumulation_steps 2 --epochs 300  --warm_start --wandb-name 'tsm_ACDC_2frame_covis_alpha_0.01_wse_20_tv_adabeta_0.75' --alpha 0.01 --warm_start_epoch 50 --beta 0.75 --save_freq 10 --load-model-motion '../../Models/cardiac_motion_baselines/ACDC/tsm_2frame_bs8_ga2/0150.pt'