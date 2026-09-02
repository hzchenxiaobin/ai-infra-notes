# Mock Interview Panel

在 CodeArts IDE（VS Code 1.96 基座）的 Webview 面板中内嵌打开模拟面试页面。

## 架构（为什么需要隧道）

目标页面 `http://47.93.85.170:5173/` 存在两个访问障碍：

1. 用户浏览器无法直接访问 47.93.85.170:5173
2. 此 CodeArts Web 构建不支持 webview `portMapping`/`asExternalUri` 端口转发，浏览器也到不了工作区的 localhost 端口

因此 v0.3.0 采用公网隧道方案：

```
浏览器
  └─> https://<random>.trycloudflare.com   (cloudflared 快速隧道，公网可达)
        └─> 工作区 0.0.0.0:15174            (node proxy.js 反向代理，含 WebSocket 透传)
              └─> http://47.93.85.170:5173/  (模拟面试 Vite 页面)
```

扩展本身只做一件事：用 iframe 内嵌打开 `mockInterview.url`（默认即隧道地址）。

## 组件

- `extension.js` / `package.json`：VS Code 扩展，注册 `打开模拟面试 (Mock Interview)` 和 `在浏览器中打开模拟面试` 两个命令
- `proxy.js`：独立反向代理，`node proxy.js [targetUrl] [port]`（默认 47.93.85.170:5173 / 15174）
- `tools/cloudflared`：Cloudflare 隧道客户端（linux/arm64 静态二进制）

## 启动后台服务（工作区重启后需重新执行）

```bash
cd vscode-mock-interview
nohup node proxy.js http://47.93.85.170:5173/ 15174 > proxy.log 2>&1 &
nohup ./tools/cloudflared tunnel --url http://127.0.0.1:15174 --no-autoupdate > cloudflared.log 2>&1 &
# 从日志中取出新的隧道地址
grep -oE "https://[a-z0-9-]+\.trycloudflare\.com" cloudflared.log
```

快速隧道的域名是**临时**的，每次重启 cloudflared 都会变化。地址变化后需要更新 VS Code 设置项
`mockInterview.url`（设置 → 搜索 mockInterview）。

## 使用

1. 安装扩展后刷新 IDE 窗口
2. `Ctrl+Shift+P` → `打开模拟面试 (Mock Interview)`

## 安装

```bash
/root/codearts/bin/codearts-server --install-extension mock-interview-panel-0.3.0.vsix
```
