// 独立反向代理：0.0.0.0:15174 -> http://47.93.85.170:5173/
// 供 cloudflared 隧道指向本端口，把面试页面暴露给浏览器。
// 用法: node proxy.js [targetUrl] [port]
const http = require('http');
const net = require('net');

const TARGET = process.argv[2] || 'http://47.93.85.170:5173/';
const PORT = Number(process.argv[3]) || 15174;

const target = new URL(TARGET);
const targetPort = Number(target.port) || (target.protocol === 'https:' ? 443 : 80);

function escapeRegExp(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

const server = http.createServer((req, res) => {
  const proxyReq = http.request(
    {
      hostname: target.hostname,
      port: targetPort,
      path: req.url,
      method: req.method,
      headers: { ...req.headers, host: target.host },
      timeout: 30000,
    },
    (proxyRes) => {
      const headers = { ...proxyRes.headers };
      // 重定向地址改回代理自身，避免浏览器拿到内部地址
      if (headers.location) {
        headers.location = headers.location
          .replace(new RegExp(escapeRegExp(target.origin), 'g'), '')
          .replace(/^\/\//, '/');
      }
      res.writeHead(proxyRes.statusCode, headers);
      proxyRes.pipe(res);
    }
  );
  proxyReq.on('timeout', () => proxyReq.destroy(new Error('target timeout')));
  proxyReq.on('error', (err) => {
    if (!res.headersSent) {
      res.writeHead(502, { 'content-type': 'text/plain; charset=utf-8' });
    }
    res.end(`代理目标不可达: ${err.message}`);
  });
  req.pipe(proxyReq);
});

// WebSocket 透传（Vite HMR 等）
server.on('upgrade', (req, socket, head) => {
  const upstream = net.connect(targetPort, target.hostname, () => {
    const lines = [`${req.method} ${req.url} HTTP/1.1`];
    for (let i = 0; i < req.rawHeaders.length; i += 2) {
      lines.push(`${req.rawHeaders[i]}: ${req.rawHeaders[i + 1]}`);
    }
    upstream.write(lines.join('\r\n') + '\r\n\r\n');
    if (head && head.length) upstream.write(head);
    upstream.pipe(socket);
    socket.pipe(upstream);
  });
  upstream.on('error', () => socket.destroy());
  socket.on('error', () => upstream.destroy());
});

server.on('error', (err) => {
  console.error('server error:', err.message);
  process.exit(1);
});

server.listen(PORT, '0.0.0.0', () => {
  console.log(`proxy listening on 0.0.0.0:${PORT} -> ${TARGET}`);
});
