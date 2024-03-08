#!/bin/bash
#SBATCH --job-name=cvpr2024_dual_uncertainty
#SBATCH --out="./slurm_out/slurm-%j.out"
#SBATCH --time=02:00:00
#SBATCH --gpus=a5000:1
#SBATCH --mem=64G
#SBATCH --partition gpu

pwd
nvidia-smi
module load miniconda
source activate torch_flow
python voxelmorph_torch/train_2frame_covis_tv_nomut.py --dataset 'CAMUS'  --bidir --model-dir '../../Models/cvpr2024_covis/CAMUS/vxm_CAMUS_2frame_covis_alpha_0.04_wse_20_tv_adabeta_1.5_nomut' --motion-loss-type 'wmse' --covis-loss-type 'coviswmse' --batch-size 8 --accumulation_steps 2 --epochs 300  --warm_start --wandb-name 'vxm_CAMUS_2frame_covis_alpha_0.04_wse_20_tv_adabeta_1.5_nomut' --alpha 0.04 --warm_start_epoch 20 --beta 1.5
