python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /vlm/yinxie/code/checkpoints/RL-qwen25-7b-VL-40kcot-20kvqa-ori \
    --target_dir /vlm/yinxie/code/checkpoints/RL-qwen25-7b-VL-40kcot-20kvqa-117steps