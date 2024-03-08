#!/bin/bash
#SBATCH --job-name=cvpr2024_dual_uncertainty
#SBATCH --out="./slurm_out/slurm-%j.out"
#SBATCH --time=04:00:00
#SBATCH --gpus=a5000:1
#SBATCH --mem=64G
#SBATCH --partition gpu

pwd
nvidia-smi
module load miniconda
source activate torch_flow
python transmorph_torch/train_2frame_adaframe.py --dataset 'ACDC' --bidir --model-dir '../../Models/cvpr2024_covis/ACDC/tsm_ACDC_2frame_adaframe' --motion-loss-type 'wmse' --batch-size 8 --epoch 150 --accumulation_steps 2 --wandb-name 'tsm_ACDC_2frame_adaframe' 
