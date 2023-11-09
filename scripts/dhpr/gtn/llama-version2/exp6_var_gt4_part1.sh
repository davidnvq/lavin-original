#!/bin/bash
#$-S /bin/bash
#$-cwd
#$-ac d=none
#$-j y
#$-o $HOME/log/exp6_var_llama2_gtn32_attn_has_box
#$ -N "exp5-1"
#$-jc gtn-container_g4.24h

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

export EXPNAME="exp6_var_llama2_gtn32_attn_has_box"
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node 4 --master_port 13320 train_dhpr.py \
    --wandb_enable \
    --llm_model llama-2-7b \
    --output_dir ./outputs/${EXPNAME} \
    --batch_size 2 \
    --accum_iter 4 \
    --visual_adapter_type router_block \
    --adapter_type attn \
    --has_boxes


export EXPNAME="exp6_var_llama2_gtn32_boxadapter_indicatormodality"
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node 4 --master_port 13320 train_dhpr.py \
    --wandb_enable \
    --llm_model llama-2-7b \
    --output_dir ./outputs/${EXPNAME} \
    --batch_size 2 \
    --accum_iter 4 \
    --visual_adapter_type router_block \
    --has_boxes \
    --adapter_type adapter_box \
    --weight_kind indicator_modality



export EXPNAME="exp6_var_llama2_gtn16_attn_has_box"
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node 4 --master_port 13320 train_dhpr.py \
    --wandb_enable \
    --llm_model llama-2-7b \
    --output_dir ./outputs/${EXPNAME} \
    --batch_size 2 \
    --accum_iter 2 \
    --visual_adapter_type router_block \
    --adapter_type attn \
    --has_boxes


export EXPNAME="exp6_var_llama2_gtn16_boxadapter_indicatormodality"
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node 4 --master_port 13320 train_dhpr.py \
    --wandb_enable \
    --llm_model llama-2-7b \
    --output_dir ./outputs/${EXPNAME} \
    --batch_size 2 \
    --accum_iter 2 \
    --visual_adapter_type router_block \
    --has_boxes \
    --adapter_type adapter_box \
    --weight_kind indicator_modality
