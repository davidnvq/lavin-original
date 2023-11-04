# cd ~/workspace/lavin-original
# source ~/anaconda3/etc/profile.d/conda.sh
# # then we can use conda
# conda activate lavin
echo "RUN eval"
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 --master_port 15121 eval.py \
    --ckpt_dir ./data/weights/ \
    --llm_model 7B \
    --tokenizer_path ./data/weights/tokenizer.model \
    --data_root ./data \
    --caption_file ./data/captions.json \
    --adapter_path ./data/sqa-llama-7b.pth \
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

echo "RUN Training"
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 torchrun --nproc_per_node 16 --master_port 11141 train.py \
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 --master_port 11341 train.py \
    --llm_model llama-2-7b \
    --llama_model_path ./data/weights/ \
    --data_path ./data/captions.json \
    --max_seq_len 512 \
    --batch_size 4 \
    --accum_iter 2 \
    --epochs 20 \
    --warmup_epochs 2 \
    --blr 9e-3 \
    --weight_decay 0.02 \
    --output_dir ./outputs/LaVIN-llama-2-7b-SciQA-gtn4/ \
    --adapter_type attn\
    --adapter_dim 8\
    --adapter_scale 1\
    --n_prompt 6 \
    --prompt_format QCM-ALE \
    --temperature 10.\
    --visual_adapter_type router

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 --master_port 11311 eval.py \
    --ckpt_dir ./data/weights/ \
    --llm_model llama-2-7b \
    --tokenizer_path ./data/weights/tokenizer.model \
    --data_root ./data \
    --caption_file ./data/captions.json \
    --adapter_path ./outputs/LaVIN-llama-2-7b-SciQA-gtn4/checkpoint-19.pth \
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
