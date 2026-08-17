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

/** Pull a colour out of the stylesheet so the charts follow the theme. */
function token(name, fallback) {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
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
      fill: cloudy && cloudy[i] ? token("--warn", "#cf7a12") : null,
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
  const accent = token("--accent-2", "#12c79c");
  heads.forEach((row, r) => {
    const hi = Math.max(...row, 1e-9);
    row.forEach((v, c) => {
      const a = Math.pow(v / hi, 0.7);
      svg.appendChild(el("rect", {
        x: c * cw, y: r * ch, width: Math.ceil(cw) + 0.3, height: ch - 0.4,
        fill: accent,
        "fill-opacity": a.toFixed(3),
      }));
    });
  });

  if (frame !== null && frame !== undefined) {
    svg.appendChild(el("rect", {
      x: frame * cw - 0.5, y: 0, width: cw + 1, height: H,
      fill: "none", stroke: token("--accent", "#0a9d7c"), "stroke-width": 0.9,
    }));
  }
  host.appendChild(svg);
}
