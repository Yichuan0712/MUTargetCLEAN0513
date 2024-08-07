#!/bin/bash
#SBATCH --mem 50G
#SBATCH -n 1
#SBATCH --gres gpu:A100:1

#SBATCH --error=error.log           # 错误日志文件

#SBATCH --time 2-00:00:00 #Time for the job to run
#SBATCH --job-name mutarget

##SBATCH -p xudong-gpu
##SBATCH -A xudong-lab

#SBATCH -p gpu


module load miniconda3

# Activate the Conda environment
#source activate /home/wangdu/.conda/envs/pytorch1.13.0
source activate /home/yz3qt/data/miniconda/envs/mutarget

export TORCH_HOME=/home/yz3qt/data/torch_cache/
export HF_HOME=/home/yz3qt/data/transformers_cache/

python train.py --config_path ./configs_yaml/0703config/config0703_E15_M0.3_M0.1_2530.yaml --result_path ./result_0703/ || true

python train.py --config_path ./configs_yaml/0703config/config0703_E15_M0.3_M0.1_5060.yaml --result_path ./result_0703/ || true

python train.py --config_path ./configs_yaml/0703config/config0703_E15_M0.3_M0.1_10125.yaml --result_path ./result_0703/ || true

python train.py --config_path ./configs_yaml/0703config/config0703_E15_M0.3_M0.1_20250.yaml --result_path ./result_0703/ || true

python train.py --config_path ./configs_yaml/0703config/config0703_E15_M0.3_M0.1_40500.yaml --result_path ./result_0703/ || true