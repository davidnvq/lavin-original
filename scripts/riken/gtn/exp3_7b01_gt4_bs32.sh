# For internet connection
export MY_PROXY_URL="http://10.1.10.1:8080/"
export HTTP_PROXY=$MY_PROXY_URL
export HTTPS_PROXY=$MY_PROXY_URL
export FTP_PROXY=$MY_PROXY_URL
export http_proxy=$MY_PROXY_URL
export https_proxy=$MY_PROXY_URL
export ftp_proxy=$MY_PROXY_URL

source ~/anaconda3/etc/profile.d/conda.sh
conda activate lavin

export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node 4 --master_port 12141 train.py \
    --wandb_enable \
    --llm_model 7B \
    --llama_model_path ./data/weights/ \
    --data_path ./data/captions.json \
    --max_seq_len 512 \
    --batch_size 4 \
    --accum_iter 2 \
    --epochs 20 \
    --warmup_epochs 2 \
    --blr 9e-3 \
    --weight_decay 0.02 \
    --output_dir ./outputs/exp3_7b01_gt4_bs32 \
    --adapter_type attn \
    --adapter_dim 8 \
    --adapter_scale 1 \
    --n_prompt 6 \
    --prompt_format QCM-ALE \
    --temperature 10.\
    --visual_adapter_type router

export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node 1 --master_port 12311 eval.py \
    --ckpt_dir ./data/weights/ \
    --llm_model 7B \
    --tokenizer_path ./data/weights/tokenizer.model \
    --data_root ./data \
    --caption_file ./data/captions.json \
    --adapter_path ./outputs/exp3_7b01_gt4_bs32/checkpoint-19.pth \
    --adapter_type attn \
    --adapter_dim 8 \
    --adapter_scale 1 \
    --prompt_format QCM-ALE \
    --max_batch_size 64 \
    --max_seq_len 256 \
    --split test \
    --n_prompt 6 \
    --temperature 10.\
    --visual_adapter_type router
