#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# 一键配置：在新服务器上生成 SSH Key 并自动添加到 GitHub
# ============================================================================
# 使用方式：
#   1. 把 GITHUB_TOKEN 改成你的 Personal Access Token（需要 write:public_key 权限）
#   2. chmod +x setup_github_ssh.sh
#   3. ./setup_github_ssh.sh
# ============================================================================

# -------------------- 必填：把 Token 写在这里 --------------------
GITHUB_TOKEN="${GITHUB_TOKEN:-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}"
# ----------------------------------------------------------------

SSH_KEY_TYPE="ed25519"
SSH_KEY_PATH="${HOME}/.ssh/id_${SSH_KEY_TYPE}"
SSH_KEY_COMMENT="$(whoami)@$(hostname)-auto-$(date +%Y%m%d)"

if [[ "$GITHUB_TOKEN" == "YOUR_GITHUB_TOKEN_HERE" ]]; then
    echo "❌ 错误：请先编辑脚本，把 GITHUB_TOKEN 替换成你的 GitHub Personal Access Token。"
    exit 1
fi

# 确保 .ssh 目录存在
mkdir -p "${HOME}/.ssh"
chmod 700 "${HOME}/.ssh"

# 生成 SSH Key（如果还没有）
if [[ ! -f "$SSH_KEY_PATH" ]]; then
    echo "🔑 正在生成 SSH Key: $SSH_KEY_PATH"
    ssh-keygen -t "$SSH_KEY_TYPE" -C "$SSH_KEY_COMMENT" -f "$SSH_KEY_PATH" -N ""
else
    echo "✅ SSH Key 已存在: $SSH_KEY_PATH，跳过生成"
fi

PUB_KEY_PATH="${SSH_KEY_PATH}.pub"
PUB_KEY_CONTENT="$(cat "$PUB_KEY_PATH")"

# 读取公钥标题（用 comment 作为标题）
KEY_TITLE="$(awk '{print $3}' "$PUB_KEY_PATH")"

# GitHub API 基础 URL
API_BASE="https://api.github.com"

# 验证 Token 是否有效
echo "🔍 验证 GitHub Token..."
HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
    -H "Authorization: token $GITHUB_TOKEN" \
    -H "Accept: application/vnd.github.v3+json" \
    "$API_BASE/user")

if [[ "$HTTP_STATUS" != "200" ]]; then
    echo "❌ 错误：GitHub Token 验证失败，HTTP 状态码 $HTTP_STATUS"
    echo "   请检查 Token 是否有效，以及是否具有 'write:public_key' 权限。"
    exit 1
fi

# 获取 GitHub 用户名
GITHUB_USER=$(curl -s \
    -H "Authorization: token $GITHUB_TOKEN" \
    -H "Accept: application/vnd.github.v3+json" \
    "$API_BASE/user" | grep -o '"login":"[^"]*"' | cut -d'"' -f4)
echo "👤 GitHub 用户: $GITHUB_USER"

# 检查是否已存在相同公钥
echo "🔍 检查 GitHub 上是否已有相同公钥..."
EXISTING_KEYS=$(curl -s \
    -H "Authorization: token $GITHUB_TOKEN" \
    -H "Accept: application/vnd.github.v3+json" \
    "$API_BASE/user/keys")

if echo "$EXISTING_KEYS" | grep -qF "$PUB_KEY_CONTENT"; then
    echo "✅ 该公钥已存在于 GitHub，无需重复添加"
else
    echo "📤 正在添加公钥到 GitHub..."
    RESPONSE=$(curl -s -w "\n%{http_code}" \
        -H "Authorization: token $GITHUB_TOKEN" \
        -H "Accept: application/vnd.github.v3+json" \
        -H "Content-Type: application/json" \
        -X POST \
        -d "{\"title\":\"$KEY_TITLE\",\"key\":\"$PUB_KEY_CONTENT\"}" \
        "$API_BASE/user/keys")

    HTTP_STATUS=$(echo "$RESPONSE" | tail -n1)
    BODY=$(echo "$RESPONSE" | sed '$d')

    if [[ "$HTTP_STATUS" == "201" ]]; then
        echo "✅ 公钥已成功添加到 GitHub"
    else
        echo "❌ 添加失败，HTTP 状态码 $HTTP_STATUS"
        echo "$BODY"
        exit 1
    fi
fi

# 配置 ssh config（可选：让 github.com 使用这个 key）
SSH_CONFIG="${HOME}/.ssh/config"
if [[ ! -f "$SSH_CONFIG" ]] || ! grep -q "Host github.com" "$SSH_CONFIG" 2>/dev/null; then
    echo "📝 配置 ~/.ssh/config..."
    cat >> "$SSH_CONFIG" <<EOF

Host github.com
    HostName github.com
    User git
    IdentityFile $SSH_KEY_PATH
    IdentitiesOnly yes
EOF
    chmod 600 "$SSH_CONFIG"
else
    echo "✅ ~/.ssh/config 已存在 github.com 配置，跳过"
fi

# 测试连接
echo "🧪 测试 SSH 连接 GitHub..."
if ssh -o StrictHostKeyChecking=accept-new -T git@github.com 2>&1 | grep -q "successfully authenticated"; then
    echo "🎉 一切就绪，可以通过 SSH 拉取/推送 GitHub 仓库了"
else
    echo "⚠️  连接测试未返回预期结果，可能是首次连接，请稍后再试：ssh -T git@github.com"
fi

echo ""
echo "公钥文件: $PUB_KEY_PATH"
echo "私钥文件: $SSH_KEY_PATH"
