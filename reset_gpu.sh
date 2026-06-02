#!/bin/bash

# 指定要重置的显卡编号
GPU_ID=0

echo "正在重置显卡 $GPU_ID 的频率设置..."

# 1. 重置应用时钟
echo "正在重置应用时钟..."
nvidia-smi -i $GPU_ID -rac
if [ $? -ne 0 ]; then
    echo "警告：重置应用时钟失败，可能该显卡不支持或权限不足"
fi

# 2. 重置核心锁频
echo "正在重置核心频率锁定..."
nvidia-smi -i $GPU_ID -rgc
if [ $? -ne 0 ]; then
    echo "警告：重置核心频率锁定失败，可能该显卡不支持或权限不足"
fi

# 3. 重置显存锁频
echo "正在重置显存频率锁定..."
nvidia-smi -i $GPU_ID -rmc
if [ $? -ne 0 ]; then
    echo "警告：重置显存频率锁定失败，可能该显卡不支持或权限不足"
fi

# 4. 可选：关闭持久模式
echo "正在关闭显卡 $GPU_ID 的持久模式..."
nvidia-smi -i $GPU_ID -pm 0
if [ $? -ne 0 ]; then
    echo "警告：关闭持久模式失败，可能权限不足"
fi

echo "显卡 $GPU_ID 频率设置已恢复默认！"

# 可选：查看当前状态
echo "当前显卡状态："
nvidia-smi -i $GPU_ID