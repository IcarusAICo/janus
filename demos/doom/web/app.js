const promptsEl = document.getElementById("prompts");
const frameEl = document.getElementById("frame");
const ordersEl = document.getElementById("orders");
const stateEl = document.getElementById("state");
let lastState = "";

function barRow(key, p, winner) {
  const pct = Math.round((p || 0) * 100);
  return `<div class="bar ${winner ? "winner" : ""}">
    <span>${key}</span>
    <i style="transform:scaleX(${p || 0})"></i>
    <span>${pct}</span>
  </div>`;
}

function renderPrompts(answers) {
  if (!answers) { promptsEl.innerHTML = ""; return; }
  promptsEl.innerHTML = Object.entries(answers).map(([id, ans]) => {
    const probs = ans.probabilities || {};
    const rows = Object.keys(probs).length
      ? Object.entries(probs).sort((a, b) => b[1] - a[1]).slice(0, 6)
          .map(([k, p]) => barRow(k, p, k === ans.choice)).join("")
      : barRow("yes", ans.noul, ans.noul >= 0.5);
    return `<article class="prompt"><div class="q">${id}</div>${rows}</article>`;
  }).join("");
}

function applyHud(msg) {
  if (msg.jpeg) frameEl.src = "data:image/jpeg;base64," + msg.jpeg;
  document.getElementById("latency").textContent = Math.round(msg.latency_ms || 0) + " ms";
  document.getElementById("kills").textContent = msg.kills ?? 0;
  document.getElementById("health").textContent = msg.health ?? 0;
  document.getElementById("ammo").textContent = msg.ammo ?? 0;
  document.getElementById("director").textContent =
    `${msg.director_spawned ?? 0} spawned / ${msg.director_killed ?? 0} killed`;
  document.getElementById("objective").textContent = msg.objective || "";
  drawMap(msg.minimap);
  if (document.activeElement !== ordersEl) ordersEl.value = msg.standing_orders || "";
  if (document.activeElement !== stateEl && msg.state_text && msg.state_text !== lastState) {
    stateEl.value = msg.state_text;
    lastState = msg.state_text;
  }
  renderPrompts(msg.answers);
}

function drawMap(minimap) {
  const svg = document.getElementById("minimap");
  if (!svg) return;
  const rooms = (minimap && minimap.rooms) || [];
  const dests = (minimap && minimap.destinations) || [];
  const player = (minimap && minimap.player) || {};
  if (!rooms.length) { svg.innerHTML = ""; return; }
  const xs = rooms.map((r) => r.x).concat(dests.map((d) => d.x), [player.x]).filter((n) => n != null);
  const ys = rooms.map((r) => r.y).concat(dests.map((d) => d.y), [player.y]).filter((n) => n != null);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const pad = 8;
  const sx = (x) => pad + ((x - minX) / Math.max(1, maxX - minX)) * (200 - pad * 2);
  const sy = (y) => 120 - pad - ((y - minY) / Math.max(1, maxY - minY)) * (120 - pad * 2);
  const dots = rooms.map((r) =>
    `<circle cx="${sx(r.x)}" cy="${sy(r.y)}" r="2.2" class="${r.visited ? "visited" : "unseen"}"></circle>`
  ).join("");
  const marks = dests.map((d) =>
    `<circle cx="${sx(d.x)}" cy="${sy(d.y)}" r="2" class="dest"></circle>`
  ).join("");
  const px = player.x != null ? sx(player.x) : 100;
  const py = player.y != null ? sy(player.y) : 60;
  svg.innerHTML = `${dots}${marks}<circle cx="${px}" cy="${py}" r="3.4" class="you"></circle>`;
}

function connect() {
  const ws = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws");
  ws.onmessage = (ev) => applyHud(JSON.parse(ev.data));
  ws.onclose = () => setTimeout(connect, 800);
}

document.getElementById("orders-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  await fetch("/orders", { method: "POST", headers: {"content-type": "application/json"},
    body: JSON.stringify({ orders: ordersEl.value }) });
});
document.getElementById("apply-state").addEventListener("click", async () => {
  const text = stateEl.value;
  let override = text;
  try { override = JSON.parse(text); } catch { override = { _yaml: text }; }
  await fetch("/state", { method: "POST", headers: {"content-type": "application/json"},
    body: JSON.stringify({ override }) });
});
document.getElementById("clear-override").addEventListener("click", async () => {
  await fetch("/state", { method: "POST", headers: {"content-type": "application/json"},
    body: JSON.stringify({ override: null }) });
});
document.getElementById("schema-json").addEventListener("click", () =>
  fetch("/schema", { method: "POST", headers: {"content-type": "application/json"},
    body: JSON.stringify({ schema: "json" }) }));
document.getElementById("schema-yaml").addEventListener("click", () =>
  fetch("/schema", { method: "POST", headers: {"content-type": "application/json"},
    body: JSON.stringify({ schema: "yaml" }) }));

connect();
