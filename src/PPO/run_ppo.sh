# plese serve the reward model before running this script

# There are three interfaces in the reward_service, which are get_outcome_reward, get_process_reward, and get_reward.

DATASET_DIR="/path/to/processed_data"
PRETRAIN_MODEL="/path/to/Skywork-o1-Open-Llama-3.1-8B"
SAVE_DIR="/path/to/checkpoint_outcome_only"

export CUDA_VISIBLE_DEVICES=0
for FILE in $DATASET_DIR/*.jsonl; do
    FILENAME=$(basename $FILE .jsonl)
    CHECKPOINT_PATH="$SAVE_DIR/${FILENAME}_checkpoint_outcome"

   deepspeed --module openrlhf.cli.train_rft \
      --pretrain $PRETRAIN_MODEL \
      --remote_rm_url http://localhost:5000/get_reward \
       --save_path $CHECKPOINT_PATH \
      --save_steps -1 \
      --logging_steps 1 \
      --eval_steps -1 \
      --micro_train_batch_size 1 \
      --train_batch_size 10 \
      --micro_rollout_batch_size 10 \
      --rollout_batch_size 20 \
      --max_epochs 1 \
      --num_episodes 6 \
      --prompt_max_len 1024 \
      --generate_max_len 2048 \
      --zero_stage 2 \
      --bf16 \
      --use_wandb True\
      --wandb_project openrlhf_train_ppo_outcome_only \
      --actor_learning_rate 3e-5 \
      --critic_learning_rate 6e-5 \
      --init_kl_coef 0.01 \
      --prompt_data $FILE \
      --input_key massage \
      --apply_chat_template \
      --max_samples 1000000 \
      --normalize_reward \
      --gradient_checkpointing \
      --adam_offload \
      --flash_attn \
      --lora_rank 4 


done