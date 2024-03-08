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
python diffusemorph_torch/train_2frame_adareg.py  --dataset 'CAMUS' -p train -c diffusemorph_torch/config/diffuseMorph_train_2D.json  --motion-loss-type 'adareg' --batch-size 8 --epoch 150 --accumulation_steps 2 --model-dir '../../Models/cvpr2024_covis/CAMUS/dfm_CAMUS_2frame_adareg' --lambda_L 20 --gamma 1 --wandb-name 'dfm_CAMUS_2frame_adareg' --c 50