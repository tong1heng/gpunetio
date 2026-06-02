#!/bin/bash

# 指定要操作的显卡编号（你原命令中是 0）
GPU_ID=1

# 1. 开启指定显卡的持久模式（必须操作，否则频率设置可能无效）
echo "正在为显卡 $GPU_ID 开启持久模式..."
nvidia-smi -i $GPU_ID -pm 1
if [ $? -ne 0 ]; then
    echo "错误：开启持久模式失败，请检查显卡编号或权限（建议 sudo 运行）"
    exit 1
fi

# 2. 读取显卡的最大显存频率和核心频率
echo "正在读取显卡 $GPU_ID 的最大频率..."
MCLOCKS=$(nvidia-smi -i $GPU_ID --query-gpu=clocks.max.mem --format=csv,noheader,nounits)
GCLOCKS=$(nvidia-smi -i $GPU_ID --query-gpu=clocks.max.gr --format=csv,noheader,nounits)

# 检查频率读取是否成功
if [ -z "$MCLOCKS" ] || [ -z "$GCLOCKS" ]; then
    echo "错误：读取显卡频率失败，请确认显卡 $GPU_ID 存在且驱动正常"
    exit 1
fi
echo "显卡 $GPU_ID 最大显存频率：$MCLOCKS MHz，最大核心频率：$GCLOCKS MHz"

# 3. 设置应用时钟（显存,核心）
echo "正在设置应用时钟为 $MCLOCKS,$GCLOCKS..."
nvidia-smi -i $GPU_ID -ac $MCLOCKS,$GCLOCKS

# 4. 锁定核心频率
echo "正在锁定核心频率为 $GCLOCKS MHz..."
nvidia-smi -i $GPU_ID -lgc $GCLOCKS

echo "所有频率设置已完成！"
# 可选：验证最终频率
echo "当前显卡频率状态："
# nvidia-smi -i $GPU_ID --query-gpu=clocks.current.mem,clocks.current.gr --format=csv