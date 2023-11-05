#!/bin/bash
#$-S /bin/bash
#$-cwd
#$-ac d=none
#$-j y
#$-o $HOME/log/eval_exp1_var_g16_bs16
#$ -N "eval_exp1_var_g16_bs16"
#$-jc gtb-container_g1.24h


# For internet connection
export MY_PROXY_URL="http://10.1.10.1:8080/"
export HTTP_PROXY=$MY_PROXY_URL
export HTTPS_PROXY=$MY_PROXY_URL
export FTP_PROXY=$MY_PROXY_URL
export http_proxy=$MY_PROXY_URL
export https_proxy=$MY_PROXY_URL
export ftp_proxy=$MY_PROXY_URL

source ~/anaconda3/etc/profile.d/conda.sh
conda activate lavin-torch2.1

export CKPT="checkpoint-19.pth"
export EXPNAME="exp1_var_g16_bs16"

torchrun --nproc_per_node 1 --master_port 14812 eval_dhpr.py \
    --adapter_path ./outputs/${EXPNAME}/${CKPT} \
    --batch_size 4


