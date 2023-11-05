#!/bin/bash
#$-S /bin/bash
#$-cwd
#$-ac d=none
#$-j y
#$-o $HOME/log/exp8_7b02_g16_bs32_train_dc
#$ -N "exp8_7b02_g16_bs32_train_dc"
#$-jc gs-container_g16.24h


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

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export EXPNAME="exp8_7b02_g16_bs32_train_dc"

torchrun --nproc_per_node 16 --master_port 12341 train_dc.py \
    --wandb_enable \
    --llm_model llama-2-7b \
    --batch_size 1 \
    --accum_iter 2 \
    --output_dir ./outputs/${EXPNAME}

export ckpt=${EXPNAME}/checkpoint-19.pth
torchrun --nproc_per_node 1 --master_port 12342 eval.py \
    --ckpt_dir ./data/weights/ \
    --llm_model 7B \
    --tokenizer_path ./data/weights/tokenizer.model \
    --data_root ./data \
    --caption_file ./data/captions.json \
    --adapter_path ./outputs/${ckpt} \
    --adapter_type attn \
    --adapter_dim 8 \
    --adapter_scale 1 \
    --prompt_format QCM-ALE \
    --max_batch_size 64 \
    --max_seq_len 256 \
    --split test \
    --n_prompt 6 \
    --temperature 10.\
    --visual_adapter_type router \
    --generation_temperature 0.0 \
    --wandb_name ${ckpt}