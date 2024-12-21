
# There are three interfaces in the reward_service, which are get_outcome_reward, get_process_reward, and get_reward.

python -m serve_rewards_assign \
    --reward_pretrain /path/to/Skywork-o1-Open-PRM-Qwen-2.5-7B \
    --port 6006 \
    --bf16 \
    --flash_attn \
    --normalize_reward \
    --max_len 4096 \
    --batch_size 1 \
    --dataset /pass/to/the/rawdata/that/not/processed \# Use the raw data to get the ground truth label. E.g.data/training_data/chemical_calculation.json