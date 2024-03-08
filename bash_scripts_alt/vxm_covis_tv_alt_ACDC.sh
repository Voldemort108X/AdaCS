#!/bin/bash
#SBATCH --job-name=cvpr2024_covis_tv
#SBATCH --out="./slurm_out/slurm-%j.out"
#SBATCH --time=04:00:00
#SBATCH --gpus=a5000:1
#SBATCH --mem=64G
#SBATCH --partition gpu

pwd
nvidia-smi
module load miniconda
source activate torch_flow
python voxelmorph_torch/train_2frame_covis_tv_alt.py --dataset 'ACDC'  --bidir --model-dir '../../Models/cvpr2024_covis/ACDC/vxm_ACDC_2frame_covis_alpha_0.02_wse_20_tv_adabeta_1.5_alt' --motion-loss-type 'wmse' --covis-loss-type 'coviswmse' --batch-size 8 --accumulation_steps 2 --epochs 300  --warm_start --wandb-name 'vxm_ACDC_2frame_covis_alpha_0.02_wse_20_tv_adabeta_1.5_alt' --alpha 0.02 --beta 1.5  --warm_start_epoch 20 
