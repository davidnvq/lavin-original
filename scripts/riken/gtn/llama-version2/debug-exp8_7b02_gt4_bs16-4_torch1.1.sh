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
echo "Python version: $(python --version)"

# from KJ & https://github.com/huggingface/accelerate/issues/314 
export NCCL_SOCKET_IFNAME=enp12s0
export NCCL_P2P_LEVEL=NVL
export NCCL_P2P_DISABLE=1 
export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_DEBUG=INFO

# from riken warnings
export OMP_NUM_THREADS=1

echo "start training..."
export EXPNAME="debug-exp8_7b02_gt4_bs16-4_torch1.1"
export LLM=llama-2-7b
torchrun --nproc_per_node 4 --master_port 13330 train.py \
    --wandb_enable \
    --llm_model ${LLM} \
    --llama_model_path ./data/weights/ \
    --data_path ./data/captions.json \
    --max_seq_len 512 \
    --batch_size 2 \
    --accum_iter 2 \
    --epochs 20 \
    --warmup_epochs 2 \
    --blr 9e-3 \
    --weight_decay 0.02 \
    --output_dir ./outputs/${EXPNAME} \
    --adapter_type attn\
    --adapter_dim 8\
    --adapter_scale 1\
    --n_prompt 6 \
    --prompt_format QCM-ALE \
    --temperature 10.\
    --visual_adapter_type router



export EXPNAME="exp1_dhpr_7b01_gt4"
torchrun --nproc_per_node 4 --master_port 13330 train_dhpr.py \
    --wandb_enable \
    --llm_model 7B \
    --output_dir ./outputs/${EXPNAME}
    

