#!/usr/bin/env bash
# 在 ai-infra-notes 目录下运行 cannbot，循环 week1-week10、day1-day7，
# 让 cannbot 阅读对应 README.md 并将其中的 ascii 图全部替换为 svg 图片。
# 相同的消息顺序执行，一天跑完再跑下一天。

set -euo pipefail

BASE_DIR="/mnt/workspace/code/github/infra/ai-infra-notes/aiinfra/daily"

for week in $(seq 1 10); do
    for day in $(seq 1 7); do
        README="$BASE_DIR/week${week}/day${day}/README.md"
        if [[ ! -f "$README" ]]; then
            echo ">>> 跳过（文件不存在）: $README"
            continue
        fi

        MESSAGE="阅读${README}，使用/mnt/workspace/aiinfra/ai-infra-notes/aiinfra/daily/leetgpu_selection.md，为教程重新挑选合适的LeetGPU 在线题目 "
        echo
        echo "======== week${week}/day${day} ========"
        echo ">>> 发送: $MESSAGE"
        cannbot run "$MESSAGE"
    done
done

echo
echo ">>> 全部执行完毕"
