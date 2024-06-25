set -x 

# We (1) changed "--save_path" and "--pretrain" in train_dpo_llama.sh according to MicroLlama;
# (2) kept only the smallest preference dataset `openai/webgpt_comparisons`;
# (3) commented out "--flash_attn" since FlashAttn does not support V100.
read -r -d '' training_commands <<EOF
../train_dpo.py \
     --save_path ./ckpt/micro-llama \
     --save_steps -1 \
     --logging_steps 1 \
     --eval_steps -1 \
     --train_batch_size 128 \
     --micro_train_batch_size 1 \
     --pretrain keeeeenw/MicroLlama \
     --bf16 \
     --max_epochs 1 \
     --max_len 2048 \
     --zero_stage 3 \
     --beta 0.1 \
     --learning_rate 5e-7 \
     --dataset openai/webgpt_comparisons \
     --gradient_checkpointing \
     --ref_offload
EOF
     # --dataset_probs 0.72,0.08,0.12,0.08 \
     # --flash_attn [lashAttention only supports Ampere GPUs or newer]
     # --wandb [WANDB_TOKENS] or True (use wandb login command)
     # --ipo [for IPO]
     # --label_smoothing 0.1 [for cDPO]


if [[ ${1} != "slurm" ]]; then
    deepspeed $training_commands
fi
