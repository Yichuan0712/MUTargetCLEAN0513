#!/bin/bash
#SBATCH --mem 50G
#SBATCH -n 1
#SBATCH --gres gpu:A100:1

#SBATCH --error=error.log           # 错误日志文件

#SBATCH --time 1-00:00:00 #Time for the job to run
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

python train.py --config_path ./configs_yaml/0621config/config0621_E15.yaml --result_path ./result_0621_lowlr/  || true
python train.py --config_path ./configs_yaml/0621config/config0621_E16.yaml --result_path ./result_0621_lowlr/  || true
python train.py --config_path ./configs_yaml/0621config/config0621_E17.yaml --result_path ./result_0621_lowlr/  || true
python train.py --config_path ./configs_yaml/0621config/config0621_E18.yaml --result_path ./result_0621_lowlr/  || true
python train.py --config_path ./configs_yaml/0621config/config0621_E19.yaml --result_path ./result_0621_lowlr/  || true
python train.py --config_path ./configs_yaml/0621config/config0621_E20.yaml --result_path ./result_0621_lowlr/  || true

