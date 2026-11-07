#!/usr/bin/env bash

set -euo pipefail

BASE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PLAN_FILE="$BASE_DIR/README.md"
REPO_DIR="$BASE_DIR/DeepGEMM-Ascend"
OUTPUT_ROOT="$BASE_DIR/daily"

if ! command -v cannbot >/dev/null 2>&1; then
    echo "错误：找不到 cannbot，请先安装或配置 PATH。" >&2
    exit 1
fi

if [[ ! -f "$PLAN_FILE" ]]; then
    echo "错误：学习计划不存在：$PLAN_FILE" >&2
    exit 1
fi

if [[ ! -d "$REPO_DIR" ]]; then
    echo "错误：源码仓库不存在：$REPO_DIR" >&2
    exit 1
fi

for week in $(seq 1 8); do
    for day in $(seq 1 7); do
        plan_day=$(( (week - 1) * 7 + day ))
        OUTPUT_DIR="$OUTPUT_ROOT/week${week}/day${day}"
        TUTORIAL="$OUTPUT_DIR/README.md"

        if [[ "$day" -eq 7 ]]; then
            echo ">>> 跳过（机动或休息日）: week${week}/day${day}"
            continue
        fi

        if [[ -s "$TUTORIAL" && "${FORCE:-0}" != "1" ]]; then
            echo ">>> 跳过（教程已存在）: $TUTORIAL"
            continue
        fi

        mkdir -p "$OUTPUT_DIR"

        MESSAGE="请为 DeepGEMM-Ascend 学习计划编写第 ${week} 周第 ${day} 天的中文教程。

先阅读学习计划 ${PLAN_FILE}，找到“第 ${plan_day} 天”对应的内容，再根据需要阅读源码仓库 ${REPO_DIR} 中的 README、源码和测试。将完整教程写入 ${TUTORIAL}，不要修改学习计划、源码仓库或其他文件。

教程要求：
1. 只聚焦当天主题，标题包含第 ${week} 周、第 ${day} 天和当天主题。
2. 包含学习目标、预备知识、详细讲解、源码导读、实践步骤、当天产出和自测题；复盘日则包含知识梳理、综合练习和通过标准。
3. 源码导读给出仓库内的真实文件路径、关键函数或类，并结合源码解释，不要凭空编造接口、参数、硬件行为或性能数据。
4. 实践步骤应具体可执行；需要 Ascend 硬件时，同时提供不依赖硬件的静态阅读或手工演算替代任务。
5. 教程应能独立阅读，使用清晰的中文 Markdown，必要时用代码、表格或示意图辅助说明。
6. 写入后检查文件内容和路径，确保教程完整且与当天计划一致。"

        echo
        echo "======== week${week}/day${day}（学习计划第 ${plan_day} 天） ========"
        echo ">>> 生成: $TUTORIAL"
        cannbot run "$MESSAGE"

        if [[ ! -s "$TUTORIAL" ]]; then
            echo "错误：cannbot 未生成非空教程：$TUTORIAL" >&2
            exit 1
        fi
    done
done

echo
echo ">>> 全部教程执行完毕"
