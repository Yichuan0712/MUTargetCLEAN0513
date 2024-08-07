#!/bin/bash
#SBATCH --mem 50G
#SBATCH -n 1
#SBATCH --gres gpu:A100:1

#SBATCH --error=error.log           # 错误日志文件

#SBATCH --time 0-12:00:00 #Time for the job to run
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

python train.py --config_path ./configs_yaml/0617config_position_weight_false/config0614_E15_T0.2_T1_pwf.yaml --result_path ./result_0617_2/  || true
python train.py --config_path ./configs_yaml/0617config_position_weight_false/config0614_E15_T0.2_T2_pwf.yaml --result_path ./result_0617_2/  || true
python train.py --config_path ./configs_yaml/0617config_position_weight_false/config0614_E15_T0.2_T3_pwf.yaml --result_path ./result_0617_2/  || true
python train.py --config_path ./configs_yaml/0617config_position_weight_false/config0614_E15_T0.2_T4_pwf.yaml --result_path ./result_0617_2/  || true
python train.py --config_path ./configs_yaml/0617config_position_weight_false/config0614_E15_T0.2_T5_pwf.yaml --result_path ./result_0617_2/  || true
python train.py --config_path ./configs_yaml/0617config_position_weight_false/config0614_E15_T0.2_T6_pwf.yaml --result_path ./result_0617_2/  || true

