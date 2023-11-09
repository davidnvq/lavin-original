#!/bin/bash
#$-S /bin/bash
#$-cwd
#$-ac d=none
#$-j y
#$-o $HOME/log/exp5_var_gs32_attn_has_box
#$ -N "exp5-2"
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


export EXPNAME="exp5_var_gs32_boxadapter_learn"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
torchrun --nproc_per_node 16 --master_port 13320 train_dhpr.py \
    --wandb_enable \
    --llm_model 7B \
    --output_dir ./outputs/${EXPNAME} \
    --batch_size 1 \
    --accum_iter 2 \
    --visual_adapter_type router_block \
    --has_boxes \
    --adapter_type adapter_box \
    --weight_kind learn


export EXPNAME="exp5_var_gs32_boxadapter_learn_noindicator"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
torchrun --nproc_per_node 16 --master_port 13320 train_dhpr.py \
    --wandb_enable \
    --llm_model 7B \
    --output_dir ./outputs/${EXPNAME} \
    --batch_size 1 \
    --accum_iter 2 \
    --visual_adapter_type router_block \
    --has_boxes \
    --adapter_type adapter_box \
    --weight_kind learn \
    --no_indicator

export EXPNAME="exp5_var_gs16_boxadapter_learn"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
torchrun --nproc_per_node 16 --master_port 13320 train_dhpr.py \
    --wandb_enable \
    --llm_model 7B \
    --output_dir ./outputs/${EXPNAME} \
    --batch_size 1 \
    --accum_iter 1 \
    --visual_adapter_type router_block \
    --has_boxes \
    --adapter_type adapter_box \
    --weight_kind learn


export EXPNAME="exp5_var_gs16_boxadapter_learn_noindicator"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
torchrun --nproc_per_node 16 --master_port 13320 train_dhpr.py \
    --wandb_enable \
    --llm_model 7B \
    --output_dir ./outputs/${EXPNAME} \
    --batch_size 1 \
    --accum_iter 1 \
    --visual_adapter_type router_block \
    --has_boxes \
    --adapter_type adapter_box \
    --weight_kind learn \
    --no_indicator