#!/bin/bash
#$-S /bin/bash
#$-cwd
#$-ac d=none
#$-j y
#$-o $HOME/log/eval_exp1_var_g16_bs32
#$ -N "eval_exp1_var_g16_bs32"
#$-jc gtb-container_g1.24h


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

export CKPT="checkpoint-19.pth"
export EXPNAME="exp1_var_g16_bs32"

torchrun --nproc_per_node 1 --master_port 16813 eval_dhpr.py \
    --adapter_path ./outputs/${EXPNAME}/${CKPT} \
    --batch_size 4


