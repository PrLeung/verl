#!/bin/bash

# 要检测的进程 PID
PID=2184093

# 要执行的脚本
SCRIPT="/vlm/peirouliang/verl/examples/grpo_trainer/run_qwen2_5_vl-7b_cot_40k_vqa_20k.sh"

# 检查进程是否存在
if ps -p $PID > /dev/null 2>&1; then
    echo "进程 $PID 仍在运行，等待其结束..."
    # 等待进程结束
    while ps -p $PID > /dev/null 2>&1; do
        sleep 600
    done
fi

echo "进程 $PID 已结束，启动脚本: $SCRIPT"
bash "$SCRIPT"
