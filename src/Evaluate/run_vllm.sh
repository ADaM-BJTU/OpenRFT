 vllm serve /path/to/Skywork-o1-Open-Llama-3.1-8B \
    --max-model-len 8192 \
    --disable-log-requests \
    --tensor-parallel-size 1 \
    --served-model-name Skywork-o1-Open-Llama-3.1-8B \
    --gpu-memory-utilization 0.95 \
    --port 8122\
    --enable-prefix-caching \

# if you want to use lora, you need to add the following parameters
    # --enable-lora \
    # --lora-modules sft-lora=/path/to/lora \
    
    
    
