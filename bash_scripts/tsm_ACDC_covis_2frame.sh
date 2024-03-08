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
python transmorph_torch/train_2frame_covis.py --dataset 'ACDC'  --bidir --model-dir '../../Models/cvpr2024_covis/ACDC/tsm_ACDC_2frame_covis_alpha_0.01_wse_20_mom_tv' --motion-loss-type 'wmse' --covis-loss-type 'coviswmse' --batch-size 8 --accumulation_steps 2 --epochs 300  --warm_start --wandb-name 'tsm_ACDC_2frame_covis_alpha_0.01_wse_20_mom_tv' --alpha 0.01 --warm_start_epoch 20
