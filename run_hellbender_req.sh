#!/bin/bash
#SBATCH --mem 50G
#SBATCH -n 1
#SBATCH --gres gpu:A100:1

#SBATCH --output=output.log         # 标准输出和错误日志文件
#SBATCH --error=error.log           # 错误日志文件
#SBATCH --time=0-00:05:00           # 设置作业运行时间限制为5分钟

#SBATCH --job-name mutarget
##SBATCH -p gpu
##SBATCH -p xudong-gpu
##SBATCH -A xudong-lab


module load miniconda3

# Activate the Conda environment
#source activate /home/wangdu/.conda/envs/pytorch1.13.0
source activate /home/yz3qt/data/miniconda/envs/mutarget

export TORCH_HOME=/home/yz3qt/data/torch_cache/
export HF_HOME=/home/yz3qt/data/transformers_cache/

#python train.py --config_path ./configs/config_supcon_onlysampling.yaml \
#--resume_path /cluster/pixstor/xudong-lab/duolin/MUTargetCLEAN/results_supcon_hardTrue_onlysampling/b10_p2_n4/2024-03-28__11-36-17/checkpoints//best_model.pth \
#--result_path ./results_supcon_hardTrue_onlysampling/b10_p2_n4/run2

python train.py --config_path ./configs/config_nosupcon_CNNlinear_yichuan.yaml \
--result_path ./result_nosupcon/cnnlinear_decoder/cnn2linear_c32_k7_drop0.15
#1594176