#!/bin/bash
#$-S /bin/bash
#$-cwd
#$-ac d=none
#$-j y
#$-o $HOME/log/eval_exp3_var_gs16_routerblk_attn
#$ -N "eval_exp3_var_gs16_routerblk_attn"
#$-jc gs-container_g16.24h

# For internet connection
export MY_PROXY_URL="http://10.1.10.1:8080/"
export HTTP_PROXY=$MY_PROXY_URL
export HTTPS_PROXY=$MY_PROXY_URL
export FTP_PROXY=$MY_PROXY_URL
export http_proxy=$MY_PROXY_URL
export https_proxy=$MY_PROXY_URL
export ftp_proxy=$MY_PROXY_URL
export JAVA_HOME=$HOME/.jre/jdk-11.0.21+9-jre
export PATH=$PATH:$JAVA_HOME/bin

source ~/anaconda3/etc/profile.d/conda.sh
conda activate lavin-torch2.1

export EXPNAME="exp3_var_gs16_routerblk_attn"
export CUDA_VISIBLE_DEVICES=0
torchrun --nproc_per_node 1 --master_port 12120 eval_dhpr.py \
    --adapter_path ./outputs/${EXPNAME}/checkpoint-19.pth
