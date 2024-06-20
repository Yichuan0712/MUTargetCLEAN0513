run_hellbender_0620_mut3.sh#!/bin/bash
#SBATCH --mem 50G
#SBATCH -n 1
#SBATCH --gres gpu:A100:1

#SBATCH --error=error.log           # 错误日志文件

#SBATCH --time 0-12:00:00 #Time for the job to run
#SBATCH --job-name mutarget

#SBATCH -p xudong-gpu
#SBATCH -A xudong-lab

##SBATCH -p gpu


module load miniconda3

# Activate the Conda environment
#source activate /home/wangdu/.conda/envs/pytorch1.13.0
source activate /home/yz3qt/data/miniconda/envs/mutarget

export TORCH_HOME=/home/yz3qt/data/torch_cache/
export HF_HOME=/home/yz3qt/data/transformers_cache/

python train.py --config_path ./configs_yaml/0620config_mut/config0620_E17_M0.3_M0.1.yaml --result_path ./result_0620_mut/  || true
python train.py --config_path ./configs_yaml/0620config_mut/config0620_E17_M0.18_M0.06.yaml --result_path ./result_0620_mut/  || true
python train.py --config_path ./configs_yaml/0620config_mut/config0620_E17_M0.21_M0.07.yaml --result_path ./result_0620_mut/  || true
python train.py --config_path ./configs_yaml/0620config_mut/config0620_E17_M0.24_M0.08.yaml --result_path ./result_0620_mut/  || true
python train.py --config_path ./configs_yaml/0620config_mut/config0620_E17_M0.27_M0.09.yaml --result_path ./result_0620_mut/  || true
python train.py --config_path ./configs_yaml/0620config_mut/config0620_E17_M0.33_M0.11.yaml --result_path ./result_0620_mut/  || true
python train.py --config_path ./configs_yaml/0620config_mut/config0620_E17_M0.36_M0.12.yaml --result_path ./result_0620_mut/  || true
