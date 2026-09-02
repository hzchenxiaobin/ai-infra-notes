// Mock Interview Panel - 在 CodeArts IDE Webview 中内嵌打开模拟面试页面
//
// v0.3.0 架构说明：
// 用户浏览器无法直接访问 http://47.93.85.170:5173/，也无法访问工作区的任何 localhost 端口
// （此 CodeArts Web 构建不支持 webview portMapping/asExternalUri 端口转发）。
// 因此采用公网隧道方案：
//   浏览器 -> https://<random>.trycloudflare.com (cloudflared 快速隧道)
//          -> 工作区 127.0.0.1:15174 (node proxy.js 反向代理)
//          -> http://47.93.85.170:5173/
// 扩展只负责用 iframe 内嵌打开 mockInterview.url（默认即隧道地址）。

const vscode = require('vscode');

const DEFAULT_URL = 'https://builds-possibilities-automobiles-marijuana.trycloudflare.com/';

function getTargetUrl() {
  const url = vscode.workspace
    .getConfiguration('mockInterview')
    .get('url', DEFAULT_URL);
  return url || DEFAULT_URL;
}

function activate(context) {
  context.subscriptions.push(
    vscode.commands.registerCommand('mockInterview.open', () => {
      openPanel();
    }),
    vscode.commands.registerCommand('mockInterview.openExternal', () => {
      vscode.env.openExternal(vscode.Uri.parse(getTargetUrl()));
    })
  );
}

function openPanel() {
  const url = getTargetUrl();

  const panel = vscode.window.createWebviewPanel(
    'mockInterview',
    '模拟面试',
    vscode.ViewColumn.One,
    {
      enableScripts: true,
      retainContextWhenHidden: true,
    }
  );

  panel.webview.html = buildHtml(url);

  panel.webview.onDidReceiveMessage((msg) => {
    if (msg && msg.type === 'openExternal') {
      vscode.env.openExternal(vscode.Uri.parse(url));
    }
  });
}

function buildHtml(url) {
  return /* html */ `<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <meta http-equiv="Content-Security-Policy"
        content="default-src 'none'; frame-src http: https:; style-src 'unsafe-inline'; script-src 'unsafe-inline';" />
  <title>模拟面试</title>
  <style>
    html, body { margin: 0; padding: 0; height: 100%; overflow: hidden; }
    body { display: flex; flex-direction: column; }
    .toolbar {
      display: flex; align-items: center; gap: 8px;
      padding: 4px 10px;
      font-family: var(--vscode-font-family, sans-serif);
      font-size: 12px;
      background: var(--vscode-editor-background, #1e1e1e);
      color: var(--vscode-foreground, #ccc);
      border-bottom: 1px solid var(--vscode-panel-border, #333);
      flex: 0 0 auto;
    }
    .toolbar .addr {
      flex: 1;
      overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
      opacity: 0.8;
    }
    .toolbar button {
      background: var(--vscode-button-background, #0e639c);
      color: var(--vscode-button-foreground, #fff);
      border: none; border-radius: 2px;
      padding: 3px 10px; cursor: pointer; font-size: 12px;
    }
    .toolbar button:hover {
      background: var(--vscode-button-hoverBackground, #1177bb);
    }
    iframe { flex: 1; width: 100%; border: none; }
  </style>
</head>
<body>
  <div class="toolbar">
    <button id="reload">刷新</button>
    <span class="addr">${url}</span>
    <button id="openExternal">在浏览器中打开</button>
  </div>
  <iframe id="frame" src="${url}" sandbox="allow-scripts allow-same-origin allow-forms allow-popups allow-downloads"></iframe>
  <script>
    const vscode = acquireVsCodeApi();
    document.getElementById('reload').addEventListener('click', () => {
      const f = document.getElementById('frame');
      f.src = f.src;
    });
    document.getElementById('openExternal').addEventListener('click', () => {
      vscode.postMessage({ type: 'openExternal' });
    });
  </script>
</body>
</html>`;
}

function deactivate() {}

module.exports = { activate, deactivate };
