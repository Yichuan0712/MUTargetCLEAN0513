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

python train.py --config_path ./configs_yaml/0618config/config0618_E15_T0.2_T5_addoriginal.yaml --result_path ./result_0618/  || true

python train.py --config_path ./configs_yaml/0618config/config0618_E15_T0.2_T5_droprate0.3.yaml --result_path ./result_0618/  || true

python train.py --config_path ./configs_yaml/0618config/config0618_E15_T0.2_T5_k5.yaml --result_path ./result_0618/  || true
python train.py --config_path ./configs_yaml/0618config/config0618_E15_T0.2_T5_k9.yaml --result_path ./result_0618/  || true
python train.py --config_path ./configs_yaml/0618config/config0618_E15_T0.2_T5_k11.yaml --result_path ./result_0618/  || true

python train.py --config_path ./configs_yaml/0618config/config0618_E15_T0.2_T5_mr1.yaml --result_path ./result_0618/  || true
python train.py --config_path ./configs_yaml/0618config/config0618_E15_T0.2_T5_mr2.yaml --result_path ./result_0618/  || true

python train.py --config_path ./configs_yaml/0618config/config0618_E15_T0.2_T5_pertimes3.yaml --result_path ./result_0618/  || true
python train.py --config_path ./configs_yaml/0618config/config0618_E15_T0.2_T5_pertimes4.yaml --result_path ./result_0618/  || true
python train.py --config_path ./configs_yaml/0618config/config0618_E15_T0.2_T5_pertimes6.yaml --result_path ./result_0618/  || true
python train.py --config_path ./configs_yaml/0618config/config0618_E15_T0.2_T5_pertimes7.yaml --result_path ./result_0618/  || true

python train.py --config_path ./configs_yaml/0618config/config0618_E15_T0.2_T5.yaml --result_path ./result_0618/  || true

