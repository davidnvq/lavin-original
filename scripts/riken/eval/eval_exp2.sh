#!/bin/bash
#$-S /bin/bash
#$-cwd
#$-ac d=none
#$-j y
#$-o $HOME/log/$JOB_ID
#$ -N eval
#$-jc gtn-container_g1.24h

export MY_PROXY_URL="http://10.1.10.1:8080/"
export HTTP_PROXY=$MY_PROXY_URL
export HTTPS_PROXY=$MY_PROXY_URL
export FTP_PROXY=$MY_PROXY_URL
export http_proxy=$MY_PROXY_URL
export https_proxy=$MY_PROXY_URL
export ftp_proxy=$MY_PROXY_URL

source ~/anaconda3/etc/profile.d/conda.sh
conda activate lavin

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 --master_port 14571 eval.py \
    --ckpt_dir ./data/weights/ \
    --llm_model 7B \
    --tokenizer_path ./data/weights/tokenizer.model \
    --data_root ./data \
    --caption_file ./data/captions.json \
    --adapter_path /home/quang/workspace/lavin-original/outputs/exp2_7b01_g16_bs32/checkpoint-19.pth \
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
    --output_dir ./outputs/ \
    --generation_temperature 0.0 \
    --wandb_name eval_exp2-ckpt19
