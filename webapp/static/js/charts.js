/* Hand-drawn SVG charts. No plotting library -- every chart here needs a real
 * calendar on the x axis, so points are placed by the day they were taken and
 * not by their position in the list. Gaps in the season stay visible as gaps.
 */

const NS = "http://www.w3.org/2000/svg";

function el(tag, attrs = {}, kids = []) {
  const node = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v !== null && v !== undefined) node.setAttribute(k, v);
  }
  for (const kid of [].concat(kids)) {
    node.appendChild(typeof kid === "string" ? document.createTextNode(kid) : kid);
  }
  return node;
}

const MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];

/** Month boundaries inside the observed window, for the vertical guide lines. */
function monthTicks(dates) {
  if (!dates.length) return [];
  const first = new Date(dates[0]);
  const last = new Date(dates[dates.length - 1]);
  const ticks = [];
  const cursor = new Date(first.getFullYear(), first.getMonth(), 1);
  while (cursor <= last) {
    if (cursor >= first) {
      ticks.push({
        t: cursor.getTime(),
        label: MONTHS[cursor.getMonth()] + (cursor.getMonth() === 0 ? " " + String(cursor.getFullYear()).slice(2) : ""),
      });
    }
    cursor.setMonth(cursor.getMonth() + 1);
  }
  return ticks;
}

function makeScale(dates, x0, x1) {
  const times = dates.map((d) => new Date(d).getTime());
  const lo = Math.min(...times);
  const hi = Math.max(...times);
  const span = Math.max(hi - lo, 1);
  return {
    times,
    at: (t) => x0 + ((t - lo) / span) * (x1 - x0),
    lo,
    hi,
  };
}

/* ─────────────────  Greenness (NDVI) through the season  ───────────────── */

export function ndviChart(host, { dates, series, classes, frame }) {
  host.textContent = "";
  if (!series || !series.length) {
    host.innerHTML = '<p class="hint">Not enough of any single crop in this patch to plot a curve.</p>';
    return;
  }

  const W = 380, H = 168, L = 30, R = 6, T = 8, B = 20;
  const sx = makeScale(dates, L, W - R);
  const all = series.flatMap((s) => s.values);
  const lo = Math.min(...all, 0);
  const hi = Math.max(...all, 0.4);
  const pad = (hi - lo) * 0.12 || 0.1;
  const yLo = lo - pad, yHi = hi + pad;
  const sy = (v) => H - B - ((v - yLo) / (yHi - yLo)) * (H - T - B);

  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" });

  for (const g of [yLo, (yLo + yHi) / 2, yHi]) {
    svg.appendChild(el("line", { class: "grid-line", x1: L, x2: W - R, y1: sy(g), y2: sy(g) }));
    svg.appendChild(el("text", { class: "axis-text", x: L - 5, y: sy(g) + 3, "text-anchor": "end" }, g.toFixed(2)));
  }

  for (const m of monthTicks(dates)) {
    const x = sx.at(m.t);
    if (x < L || x > W - R) continue;
    svg.appendChild(el("line", { class: "grid-line", x1: x, x2: x, y1: T, y2: H - B }));
    svg.appendChild(el("text", { class: "axis-text", x, y: H - 6, "text-anchor": "middle" }, m.label));
  }

  for (const s of series) {
    const pts = s.values
      .map((v, i) => `${sx.at(sx.times[i]).toFixed(1)},${sy(v).toFixed(1)}`)
      .join(" ");
    svg.appendChild(
      el("polyline", { class: "series-line", points: pts, stroke: classes[s.class_id].color })
    );
  }

  if (frame !== null && frame !== undefined && sx.times[frame] !== undefined) {
    const x = sx.at(sx.times[frame]);
    svg.appendChild(el("line", { class: "now-line", x1: x, x2: x, y1: T, y2: H - B }));
  }

  svg.appendChild(el("line", { class: "axis-line", x1: L, x2: W - R, y1: H - B, y2: H - B }));
  host.appendChild(svg);

  const legend = document.createElement("div");
  legend.className = "chart-legend";
  for (const s of series) {
    const item = document.createElement("span");
    item.innerHTML =
      `<i style="background:${classes[s.class_id].color}"></i>${classes[s.class_id].name}`;
    legend.appendChild(item);
  }
  host.appendChild(legend);
}

/* ─────────────────  How much each date mattered  ───────────────── */

export function attentionChart(host, { dates, weights, frame, cloudy, onPick }) {
  host.textContent = "";
  const W = 380, H = 120, L = 30, R = 6, T = 8, B = 20;
  const sx = makeScale(dates, L, W - R);
  const hi = Math.max(...weights, 1e-9);
  const barW = Math.max(2, (W - L - R) / Math.max(dates.length, 1) * 0.62);

  const svg = el("svg", { viewBox: `0 0 ${W} ${H}` });

  for (const g of [0, 0.5, 1]) {
    const y = H - B - g * (H - T - B);
    svg.appendChild(el("line", { class: "grid-line", x1: L, x2: W - R, y1: y, y2: y }));
    svg.appendChild(el("text", { class: "axis-text", x: L - 5, y: y + 3, "text-anchor": "end" },
      g === 0 ? "0" : g === 1 ? "max" : ""));
  }

  for (const m of monthTicks(dates)) {
    const x = sx.at(m.t);
    if (x < L || x > W - R) continue;
    svg.appendChild(el("text", { class: "axis-text", x, y: H - 6, "text-anchor": "middle" }, m.label));
  }

  weights.forEach((w, i) => {
    const h = Math.max(1, (w / hi) * (H - T - B));
    const x = sx.at(sx.times[i]) - barW / 2;
    const bar = el("rect", {
      class: "att-bar" + (i === frame ? " is-on" : ""),
      x, y: H - B - h, width: barW, height: h, rx: 1,
      fill: cloudy && cloudy[i] ? "#e5a95c" : null,
    });
    bar.appendChild(el("title", {}, `${dates[i]} — weight ${(w / hi * 100).toFixed(0)}% of max`));
    bar.addEventListener("click", () => onPick && onPick(i));
    svg.appendChild(bar);
  });

  svg.appendChild(el("line", { class: "axis-line", x1: L, x2: W - R, y1: H - B, y2: H - B }));
  host.appendChild(svg);
}

/* ─────────────────  All 16 heads at once  ───────────────── */

export function headsStrip(host, { heads, frame }) {
  host.textContent = "";
  if (!heads || !heads.length) return;
  const rows = heads.length;
  const cols = heads[0].length;
  const cw = 380 / cols;
  const ch = 4.2;
  const H = rows * ch;
  const svg = el("svg", { viewBox: `0 0 380 ${H}` });

  // Each head is normalised on its own: what matters is where a head put its
  // weight, not whether one head is louder overall than another.
  heads.forEach((row, r) => {
    const hi = Math.max(...row, 1e-9);
    row.forEach((v, c) => {
      const a = Math.pow(v / hi, 0.7);
      svg.appendChild(el("rect", {
        x: c * cw, y: r * ch, width: Math.ceil(cw) + 0.3, height: ch - 0.4,
        fill: `rgba(79,214,184,${a.toFixed(3)})`,
      }));
    });
  });

  if (frame !== null && frame !== undefined) {
    svg.appendChild(el("rect", {
      x: frame * cw - 0.5, y: 0, width: cw + 1, height: H,
      fill: "none", stroke: "#4fd6b8", "stroke-width": 0.8,
    }));
  }
  host.appendChild(svg);
}
