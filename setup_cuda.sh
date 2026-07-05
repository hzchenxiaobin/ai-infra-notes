#!/bin/bash
# 在 WSL Ubuntu 上安装 CUDA Toolkit 13.3（含 ncu、nvcc、nsys）
# 用法: sudo bash setup_cuda.sh
# 说明: RTX 5090 (Blackwell, sm_120) 需要 CUDA 12.8+，Ubuntu 26.04 仓库提供 13.3。

set -euo pipefail

UBUNTU_VERSION="ubuntu2604"
CUDA_PKG="cuda-toolkit-13-3"
KEYRING_DEB="cuda-keyring_1.1-1_all.deb"
KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/${UBUNTU_VERSION}/x86_64/${KEYRING_DEB}"

# 1. 安装依赖
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y wget gnupg software-properties-common build-essential

# 2. 添加 NVIDIA CUDA 仓库 keyring
TMPDIR=$(mktemp -d)
cd "$TMPDIR"
wget -q "$KEYRING_URL"
dpkg -i "$KEYRING_DEB"
apt-get update

# 3. 安装 CUDA Toolkit 13.3（仅工具包，不安装驱动；WSL 驱动由 Windows 宿主提供）
apt-get install -y "$CUDA_PKG"

# 4. 设置系统级环境变量
CUDA_PATH="/usr/local/cuda"
cat > /etc/profile.d/cuda.sh <<EOF
export CUDA_HOME=${CUDA_PATH}
export PATH=${CUDA_PATH}/bin:\$PATH
export LD_LIBRARY_PATH=${CUDA_PATH}/lib64:\$LD_LIBRARY_PATH
EOF
chmod 644 /etc/profile.d/cuda.sh

# 5. 清理
cd /
rm -rf "$TMPDIR"

echo "========================================"
echo "CUDA Toolkit 13.3 安装完成"
echo "请重新打开终端或执行: source /etc/profile.d/cuda.sh"
echo "验证命令: ncu --version && nvcc --version && nsys --version"
echo "========================================"
