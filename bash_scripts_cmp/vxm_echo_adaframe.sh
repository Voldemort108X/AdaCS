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
python voxelmorph_torch/train_2frame_adaframe.py --dataset 'Echo' --bidir --model-dir '../../Models/cvpr2024_covis/Echo/vxm_echo_2frame_adaframe' --motion-loss-type 'wmse' --batch-size 8 --epoch 150 --accumulation_steps 2 --wandb-name 'vxm_echo_2frame_adaframe' 