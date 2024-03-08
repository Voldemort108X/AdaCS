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
python transmorph_torch/train_2frame_covis_tv_alt.py --dataset 'CAMUS'  --bidir --model-dir '../../Models/cvpr2024_covis/CAMUS/tsm_CAMUS_2frame_covis_alpha_0.03_lmbd_0.03_wse_20_tv_adabeta_5_alt' --motion-loss-type 'wmse' --covis-loss-type 'coviswmse' --batch-size 8 --accumulation_steps 2 --epochs 300 --lambda 0.03 --warm_start --wandb-name 'tsm_CAMUS_2frame_covis_alpha_0.03_lmbd_0.03_wse_20_tv_adabeta_5_alt' --alpha 0.03 --warm_start_epoch 20 --beta 5
