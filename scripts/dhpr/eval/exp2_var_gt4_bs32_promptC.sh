#!/bin/bash
#$-S /bin/bash
#$-cwd
#$-ac d=none
#$-j y
#$-o $HOME/log/eval_exp2_var_gt4_bs32_promptC
#$ -N "eval_exp2_var_gt4_bs32_promptC"
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
export EXPNAME="exp2_var_gt4_bs32_promptC"

torchrun --nproc_per_node 1 --master_port 19012 eval_dhpr.py \
    --adapter_path ./outputs/${EXPNAME}/${CKPT} \
    --batch_size 4


