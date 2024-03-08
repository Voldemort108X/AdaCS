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
python voxelmorph_torch/train_2frame_adareg.py --dataset 'Echo' --bidir --model-dir '../../Models/cvpr2024_covis/Echo/vxm_echo_2frame_adareg' --motion-loss-type 'mse' --batch-size 8 --epoch 150 --accumulation_steps 2 --wandb-name 'vxm_echo_2frame_adareg' --c 50 --int-downsize 1