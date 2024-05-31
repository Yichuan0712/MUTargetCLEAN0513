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

#python train0520.py --config_path ./configs/config_supcon_onlysampling.yaml \
#--resume_path /cluster/pixstor/xudong-lab/duolin/MUTargetCLEAN/results_supcon_hardTrue_onlysampling/b10_p2_n4/2024-03-28__11-36-17/checkpoints//best_model.pth \
#--result_path ./results_supcon_hardTrue_onlysampling/b10_p2_n4/run2

python train.py --config_path ./configs/config0530yichuan.yaml --result_path ./result_nosupcon/0530_new/  || true
python train.py --config_path ./configs/config0530yichuan_cnn.yaml --result_path ./result_nosupcon/0530_new/  || true
python train.py --config_path ./configs/config0530yichuan_addorigin.yaml --result_path ./result_nosupcon/0530_new/  || true
python train.py --config_path ./configs/config0530yichuan_droprate.yaml --result_path ./result_nosupcon/0530_new/  || true
python train.py --config_path ./configs/config0530yichuan_gamma.yaml --result_path ./result_nosupcon/0530_new/  || true
python train.py --config_path ./configs/config0530yichuan_pertime3.yaml --result_path ./result_nosupcon/0530_new/  || true
python train.py --config_path ./configs/config0530yichuan_kernel5.yaml --result_path ./result_nosupcon/0530_new/  || true
python train.py --config_path ./configs/config0530yichuan_kernel9.yaml --result_path ./result_nosupcon/0530_new/  || true
python train.py --config_path ./configs/config0530yichuan_lossweight.yaml --result_path ./result_nosupcon/0530_new/  || true
python train.py --config_path ./configs/config0530yichuan_noaug.yaml --result_path ./result_nosupcon/0530_new/  || true
python train.py --config_path ./configs/config0530yichuan_posweight.yaml --result_path ./result_nosupcon/0530_new/  || true
python train.py --config_path ./configs/config0530yichuan_warmup.yaml --result_path ./result_nosupcon/0530_new/  || true
