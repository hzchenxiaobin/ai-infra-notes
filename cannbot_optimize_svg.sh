#!/usr/bin/env bash
# 遍历 week1-week10 的 images 目录下的所有 svg 文件，
# 逐个发送给 cannbot，让 cannbot 优化对应的 svg。
# 相同的消息顺序执行，一个跑完再跑下一个。

set -euo pipefail

BASE_DIR="/mnt/workspace/aiinfra/ai-infra-notes/aiinfra/daily"

for week in $(seq 1 10); do
    IMAGES_DIR="$BASE_DIR/week${week}/images"
    if [[ ! -d "$IMAGES_DIR" ]]; then
        echo ">>> 跳过（目录不存在）: $IMAGES_DIR"
        continue
    fi

    echo
    echo "######## week${week} ########"

    for SVG in "$IMAGES_DIR"/*.svg; do
        if [[ ! -f "$SVG" ]]; then
            echo ">>> 跳过（无 svg 文件）: $IMAGES_DIR"
            break
        fi

        FILENAME="$(basename "$SVG")"
        MESSAGE="优化${SVG}"
        echo
        echo "======== ${FILENAME} ========"
        echo ">>> 发送: $MESSAGE"
        cannbot run "$MESSAGE"
    done
done

echo
echo ">>> 全部执行完毕"
