#!/bin/bash
#$-S /bin/bash
#$-cwd
#$-ac d=none
#$-j y
#$-o $HOME/log/exp4_var_gt4_bs32_llama2
#$ -N "llama2"
#$-jc gtb-container_g4.24h


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

export CUDA_VISIBLE_DEVICES=0,1,2,3

export EXPNAME="exp4_var_gt4_bs32_llama2"
torchrun --nproc_per_node 4 --master_port 13310 train_dhpr.py \
    --wandb_enable \
    --llm_model llama-2-7b \
    --output_dir ./outputs/${EXPNAME} \
    --batch_size 2 \
    --accum_iter 2
