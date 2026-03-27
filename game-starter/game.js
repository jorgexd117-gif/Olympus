const canvas = document.getElementById('game');
const ctx = canvas.getContext('2d');
const statusEl = document.getElementById('status');
const keys = new Set();
const player = { x: 80, y: 220, w: 26, h: 26, speed: 240 };
const enemies = Array.from({ length: 7 }, (_, i) => ({
  x: 420 + i * 65,
  y: 40 + Math.random() * 430,
  r: 12,
  vx: -130 - Math.random() * 120,
}));
let running = true;
let start = performance.now();
window.addEventListener('keydown', (e) => keys.add(e.key.toLowerCase()));
window.addEventListener('keyup', (e) => keys.delete(e.key.toLowerCase()));
function hit(a, b) {
  const cx = Math.max(a.x, Math.min(b.x, a.x + a.w));
  const cy = Math.max(a.y, Math.min(b.y, a.y + a.h));
  return ((cx - b.x) ** 2 + (cy - b.y) ** 2) < b.r ** 2;
}
function update(dt) {
  if (!running) return;
  const left = keys.has('arrowleft') || keys.has('a');
  const right = keys.has('arrowright') || keys.has('d');
  const up = keys.has('arrowup') || keys.has('w');
  const down = keys.has('arrowdown') || keys.has('s');
  if (left) player.x -= player.speed * dt;
  if (right) player.x += player.speed * dt;
  if (up) player.y -= player.speed * dt;
  if (down) player.y += player.speed * dt;
  player.x = Math.max(0, Math.min(canvas.width - player.w, player.x));
  player.y = Math.max(0, Math.min(canvas.height - player.h, player.y));
  for (const e of enemies) {
    e.x += e.vx * dt;
    if (e.x < -30) {
      e.x = canvas.width + 20 + Math.random() * 160;
      e.y = 30 + Math.random() * 470;
      e.vx = -130 - Math.random() * 120;
    }
    if (hit(player, e)) running = false;
  }
}
function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#38bdf8';
  ctx.fillRect(player.x, player.y, player.w, player.h);
  ctx.fillStyle = '#ef4444';
  for (const e of enemies) {
    ctx.beginPath();
    ctx.arc(e.x, e.y, e.r, 0, Math.PI * 2);
    ctx.fill();
  }
}
let last = performance.now();
function loop(now) {
  const dt = Math.min(0.033, (now - last) / 1000);
  last = now;
  update(dt);
  draw();
  const secs = ((now - start) / 1000).toFixed(1);
  statusEl.textContent = running ? `Survival: ${secs}s` : `Game Over. Survival: ${secs}s`;
  requestAnimationFrame(loop);
}
requestAnimationFrame(loop);
