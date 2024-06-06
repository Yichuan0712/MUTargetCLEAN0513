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

python train.py --config_path ./0603config/config0601yichuan_3_5_y.yaml --result_path ./result_nosupcon/0603_1/  || true
python train.py --config_path ./0603config/config0601yichuan_3_7_y.yaml --result_path ./result_nosupcon/0603_1/  || true
python train.py --config_path ./0603config/config0601yichuan_5_5_n_dp0.3.yaml --result_path ./result_nosupcon/0603_1/  || true
python train.py --config_path ./0603config/config0601yichuan_5_5_n_lowlr.yaml --result_path ./result_nosupcon/0603_1/  || true
python train.py --config_path ./0603config/config0601yichuan_5_7_n_dp0.3.yaml --result_path ./result_nosupcon/0603_1/  || true
python train.py --config_path ./0603config/config0601yichuan_5_7_n_lowlr.yaml --result_path ./result_nosupcon/0603_1/  || true
