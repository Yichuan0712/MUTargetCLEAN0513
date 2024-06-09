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

python train.py --config_path ./configs_yaml/0608config/config0608_15.yaml --result_path ./result_0608/  || true
python train.py --config_path ./configs_yaml/0608config/config0608_20.yaml --result_path ./result_0608/  || true
python train.py --config_path ./configs_yaml/0608config/config0608_25.yaml --result_path ./result_0608/  || true
python train.py --config_path ./configs_yaml/0608config/config0608_30.yaml --result_path ./result_0608/  || true

python train.py --config_path ./configs_yaml/0608config/config0608_15_.yaml --result_path ./result_0608/  || true
python train.py --config_path ./configs_yaml/0608config/config0608_20_.yaml --result_path ./result_0608/  || true
python train.py --config_path ./configs_yaml/0608config/config0608_25_.yaml --result_path ./result_0608/  || true
python train.py --config_path ./configs_yaml/0608config/config0608_30_.yaml --result_path ./result_0608/  || true
