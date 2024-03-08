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
python diffusemorph_torch/train_2frame_covis_tv_alt.py --dataset 'ACDC' -p train -c diffusemorph_torch/config/diffuseMorph_train_2D.json --model-dir '../../Models/cvpr2024_covis/ACDC/dfm_ACDC_2frame_covis_alpha_0.1_wse_20_tv_adabeta_0.1_alt' --motion-loss-type 'wmse' --covis-loss-type 'coviswmse' --batch-size 8 --accumulation_steps 2 --epoch 300 --lambda_L 20 --gamma 1 --warm_start --wandb-name 'dfm_ACDC_2frame_covis_alpha_0.1_wse_20_tv_adabeta_0.1_alt' --alpha 0.1 --warm_start_epoch 20 --beta 0.1
