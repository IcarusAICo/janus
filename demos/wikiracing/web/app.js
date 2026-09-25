const lanes = document.getElementById("lanes");
const clock = document.getElementById("clock");
const title = document.getElementById("title");

function render(snap) {
  if (!snap || !snap.racers) return;
  title.textContent = `${snap.start} → ${snap.target}`;
  clock.textContent = ((snap.elapsed_ms || 0) / 1000).toFixed(1) + "s";
  lanes.innerHTML = Object.values(snap.racers).map((r) => {
    const path = (r.path || []).map((p) => `<li>${p}</li>`).join("");
    const top = (r.top || []).map(([k, p]) =>
      `<div><span>${k}</span><span>${Math.round(p * 100)}</span></div>`).join("");
    const cls = ["lane", r.name === "Jev" ? "jev" : "", r.done ? "done" : ""].join(" ");
    const status = r.done ? "arrived" : (r.failed || r.stage || "running");
    return `<article class="${cls}">
      <h2>${r.name}</h2>
      <p class="page">${r.title || ""}</p>
      <ol class="path">${path}</ol>
      <div class="top">${top}</div>
      <p class="stats">${r.hops} hops · ${Math.round(r.latency_ms || 0)} ms · ${status}</p>
    </article>`;
  }).join("");
}

function connect() {
  const ws = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws");
  ws.onmessage = (ev) => render(JSON.parse(ev.data));
  ws.onclose = () => setTimeout(connect, 800);
}
document.getElementById("go").addEventListener("click", () => fetch("/start", { method: "POST" }));
connect();
