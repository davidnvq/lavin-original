#!/bin/bash
#$-S /bin/bash
#$-cwd
#$-ac d=none
#$-j y
#$-o $HOME/log/exp2_var_gt4_bs32_promptB
#$ -N "exp2_var_gt4_bs32_promptB"
#$-jc gtn-container_g4.24h


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

export EXPNAME="exp2_var_gt4_bs32_promptB"
torchrun --nproc_per_node 4 --master_port 13320 train_dhpr.py \
    --wandb_enable \
    --llm_model 7B \
    --output_dir ./outputs/${EXPNAME} \
    --batch_size 2 \
    --accum_iter 4 \
    --prompt_id B

