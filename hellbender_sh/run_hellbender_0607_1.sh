#!/bin/bash
#SBATCH --mem 50G
#SBATCH -n 1
#SBATCH --gres gpu:A100:1

#SBATCH --error=error.log           # 错误日志文件

#SBATCH --time 2-00:00:00 #Time for the job to run
#SBATCH --job-name mutarget
#SBATCH -p gpu
##SBATCH -p xudong-gpu
##SBATCH -A xudong-lab


module load miniconda3

# Activate the Conda environment
#source activate /home/wangdu/.conda/envs/pytorch1.13.0
source activate /home/yz3qt/data/miniconda/envs/mutarget

export TORCH_HOME=/home/yz3qt/data/torch_cache/
export HF_HOME=/home/yz3qt/data/transformers_cache/

python train.py --config_path ./configs_yaml/0607config/config0607yichuan_lr1e-5.yaml --result_path ./result_0607_1/  || true
python train.py --config_path ./configs_yaml/0607config/config0607yichuan_lr1e-6.yaml --result_path ./result_0607_1/  || true
python train.py --config_path ./configs_yaml/0607config/config0607yichuan_lr5e-6.yaml --result_path ./result_0607_1/  || true

python train.py --config_path ./configs_yaml/0607config/config0607yichuan_mr1.yaml --result_path ./result_0607_1/  || true
python train.py --config_path ./configs_yaml/0607config/config0607yichuan_mr2.yaml --result_path ./result_0607_1/  || true
