/* Wiring. Fetches from the API, keeps one object of state, and re-renders the
 * parts of the page that depend on whatever just changed.
 */

import { Viewer, loadImage, toGrayArray } from "./viewer.js";
import { attentionChart, headsStrip } from "./charts.js";
import { PatchMap } from "./map.js";

const $ = (id) => document.getElementById(id);

// Three ways of looking at the same patch. Only one at a time -- that is the
// whole point, and stacking them was what made "Mistakes" bleed into the
// ground-truth view.
const VIEWS = [
  { key: "pred",  name: "Prediction",   note: "model", needs: "pred" },
  { key: "truth", name: "Ground truth", note: "dataset", needs: "truth" },
  { key: "error", name: "Mistakes",     note: "pred ≠ truth", needs: "error" },
  { key: "none",  name: "Image only",   note: "no overlay" },
];

const PLAY_ICON = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5l11 7-11 7z"/></svg>';
const PAUSE_ICON = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 5h3.5v14H7zM13.5 5H17v14h-3.5z"/></svg>';

// Patch ids held at the top of the sample list, in this order.
const PINNED = [40274, 40022, 30017, 30380];

const S = {
  status: null,
  classes: [],
  result: null,
  viewer: null,
  map: null,
  playing: null,
  activeSample: null,
  samples: [],
};

/* ───────────────────────────  helpers  ─────────────────────────── */

function toast(message, bad = false) {
  const t = $("toast");
  t.textContent = message;
  t.className = "toast" + (bad ? " bad" : "");
  t.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => (t.hidden = true), bad ? 7000 : 3500);
}

function busy(on, label) {
  $("busy").hidden = !on;
  if (label) $("busy-label").textContent = label;
}

function stat(label, value, cls = "", unit = "") {
  const d = document.createElement("span");
  d.className = "stat " + cls;
  d.innerHTML = `<label></label><b><span></span><small></small></b>`;
  d.querySelector("label").textContent = label;
  d.querySelector("span").textContent = value;
  d.querySelector("small").textContent = unit;
  return d;
}

const fmt = (v, digits = 1) =>
  v === null || v === undefined || Number.isNaN(v) ? "—" : Number(v).toFixed(digits);

/* ───────────────────────────  boot  ─────────────────────────── */

async function boot() {
  let status;
  try {
    status = await (await fetch("/api/status")).json();
  } catch (e) {
    $("chip-status").textContent = "server unreachable";
    return;
  }
  S.status = status;
  S.classes = status.classes;

  const bar = $("statbar");
  bar.querySelectorAll(".stat").forEach((e) => e.remove());

  if (status.model_loaded) {
    const run = status.run;
    const test = run.test_metrics || {};
    if (test.test_accuracy !== undefined)
      bar.appendChild(stat("Accuracy", fmt(test.test_accuracy, 0), "hero", "%"));
    if (test.test_IoU !== undefined) {
      // test_IoU is written as a percentage; older runs wrote a fraction.
      const iou = test.test_IoU < 1 ? test.test_IoU * 100 : test.test_IoU;
      bar.appendChild(stat("mIoU", fmt(iou, 0), "hero"));
    }
    bar.appendChild(stat("Parameters", run.n_params.toLocaleString()));
    bar.appendChild(stat("Device", run.device));
  } else {
    bar.appendChild(stat("Model", "not loaded", "bad"));
    toast(status.error || "No checkpoint found.", true);
  }

  const epochs = (status.config || {}).epochs;
  if (epochs !== undefined && epochs !== null) bar.appendChild(stat("Epochs", epochs));
  const px = (status.patch || {}).pixel_size_m;
  if (px !== undefined && px !== null)
    bar.appendChild(stat("Resolution", fmt(px, 0), "", "m/px"));

  S.viewer = new Viewer($("canvas"), S.classes);
  S.map = new PatchMap("map");

  buildLayerList();
  wireControls();
  await loadSamples();
}

async function loadSamples() {
  let data;
  try {
    data = await (await fetch("/api/samples")).json();
  } catch (e) {
    return;
  }
  S.samples = data.samples || [];
  S.folds = data.folds || null;

  if (!S.samples.length) {
    $("samples-hint").innerHTML =
      "No patches found in <code>webapp/samples/</code>. Run " +
      "<code>notebooks/export_samples.py</code> on Kaggle and unzip the result here.";
    return;
  }

  renderSampleList();

  // The address bar remembers which patch is open, so a link to a particular
  // one can be shared or reloaded and land on the same view.
  const wanted = decodeURIComponent(location.hash.slice(1));
  const shown = visibleSamples();
  const target = shown.find((s) => s.file === wanted) || shown[0];
  if (!target) return;
  const btn = document.querySelector(`.sample[data-file="${CSS.escape(target.file)}"]`);
  if (btn) btn.click();
}

/** In render order: the pinned patches, then the rest of group 1. */
function visibleSamples() {
  // A manifest-less samples dir comes back from a bare directory scan with no
  // group field on anything; hiding on that would empty the list.
  const grouped = S.samples.some((s) => s.group !== undefined);
  const pool = grouped ? S.samples.filter((s) => s.group === 1) : S.samples;
  const pinned = PINNED.map((id) => pool.find((s) => s.patch_id === id)).filter(Boolean);
  return pinned.concat(pool.filter((s) => !pinned.includes(s)));
}

function renderSampleList() {
  const list = $("sample-list");
  list.textContent = "";
  const shown = visibleSamples();
  const nPinned = shown.filter((s) => PINNED.includes(s.patch_id)).length;
  $("samples-hint").textContent =
    `${shown.length} patch${shown.length === 1 ? "" : "es"} ready`;

  shown.forEach((s, i) => {
    const li = document.createElement("li");
    if (nPinned && i === nPinned) li.className = "after-pinned";
    const btn = document.createElement("button");
    btn.className = "sample" + (s.file === S.activeSample ? " is-on" : "");
    btn.dataset.file = s.file;
    const sub = [
      s.n_dates ? `${s.n_dates} dates` : null,
      s.n_classes ? `${s.n_classes} crops` : null,
      s.size_mb ? `${s.size_mb} MB` : null,
    ].filter(Boolean).join(" · ");
    btn.innerHTML = `<span class="sample-id"></span>
      <span class="sample-meta"><b></b><span></span></span>`;
    btn.querySelector(".sample-id").textContent = s.patch_id ? "#" + s.patch_id : "npz";
    btn.querySelector("b").textContent = s.label || s.name;
    btn.querySelector(".sample-meta span").textContent = sub;
    btn.addEventListener("click", () => runSample(s, btn));
    li.appendChild(btn);
    list.appendChild(li);
  });
}

/* ───────────────────────────  running the model  ─────────────────────────── */

async function runSample(entry, btn) {
  const body = new FormData();
  body.append("sample", entry.file);
  document.querySelectorAll(".sample").forEach((b) => b.classList.remove("is-on"));
  if (btn) btn.classList.add("is-on");
  S.activeSample = entry.file;
  S.activeEntry = entry;
  history.replaceState(null, "", "#" + encodeURIComponent(entry.file));
  await run(body, entry.file);
}

async function run(body, label) {
  busy(true, "Reading the year and running U-TAE…");
  try {
    const res = await fetch("/api/predict", { method: "POST", body });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || "prediction failed");
    }
    const payload = await res.json();
    busy(true, "Drawing the maps…");
    await present(payload);
  } catch (e) {
    toast(e.message, true);
  } finally {
    busy(false);
  }
}

async function present(p) {
  const base = `/api/map/${p.token}/`;
  const wanted = ["sheet", "pred", "conf", "truth", "error"].filter((k) =>
    p.layers.includes(k)
  );
  const imgs = {};
  await Promise.all(
    wanted.map(async (k) => {
      imgs[k] = await loadImage(base + k + ".png");
    })
  );

  const data = {
    sheet: imgs.sheet,
    pred: imgs.pred ? toGrayArray(imgs.pred) : null,
    truth: imgs.truth ? toGrayArray(imgs.truth) : null,
    conf: imgs.conf ? toGrayArray(imgs.conf) : null,
    error: imgs.error ? toGrayArray(imgs.error) : null,
    attention: p.attention_space,
    attentionShape: p.attention_shape,
  };

  S.result = p;
  stopPlay();
  S.viewer.setData(data);
  S.viewer.state.mode = "pred";
  S.viewer.state.uncertainty = false;
  S.viewer.state.solo = null;

  $("empty-state").hidden = true;
  $("viewer-shell").hidden = false;
  $("panel").hidden = false;
  $("layers-card").hidden = false;
  $("legend-card").hidden = false;
  $("filmstrip-card").hidden = false;

  $("patch-title").textContent = p.patch_id ? `Patch ${p.patch_id}` : p.name;
  $("patch-sub").textContent = [
    `${p.n_dates} acquisitions`,
    p.tile || null,
    "128 × 128 px · 10 m · 163.84 ha",
  ].filter(Boolean).join("  ·  ");

  const slider = $("date-slider");
  slider.max = p.n_dates - 1;
  slider.value = 0;

  buildLayerList();
  buildLegend();
  buildKpis();
  buildNotes();
  buildTable();
  buildTimeline();
  buildFilmstrip();
  setFrame(0);
  await showMap();
}

/* ───────────────────────────  left rail  ─────────────────────────── */

function buildLayerList() {
  const host = $("layer-list");
  host.textContent = "";
  const p = S.result;
  const v = S.viewer;

  for (const def of VIEWS) {
    const missing = def.needs && p && !p.layers.includes(def.needs);
    const on = v ? v.state.mode === def.key : def.key === "pred";
    const li = document.createElement("li");
    li.className =
      "layer " + (on ? "is-on" : "is-off") + (missing || !p ? " is-disabled" : "");
    li.innerHTML = `<span class="radio"></span>
      <span class="layer-name"></span><span class="layer-note"></span>`;
    li.querySelector(".layer-name").textContent = def.name;
    li.querySelector(".layer-note").textContent = missing ? "no labels" : def.note;
    li.addEventListener("click", () => {
      v.set("mode", def.key);
      buildLayerList();
      positionWipe();
    });
    host.appendChild(li);
  }

  const sep = document.createElement("li");
  sep.className = "layer-sep";
  host.appendChild(sep);

  const on = v ? v.state.uncertainty : false;
  const li = document.createElement("li");
  li.className = "layer " + (on ? "is-on" : "is-off") + (p ? "" : " is-disabled");
  li.innerHTML = `<span class="switch"></span>
    <span class="layer-name">Uncertainty</span><span class="layer-note">pink = unsure</span>`;
  li.addEventListener("click", () => {
    v.set("uncertainty", !v.state.uncertainty);
    buildLayerList();
    positionWipe();
  });
  host.appendChild(li);
}

function buildLegend() {
  const host = $("legend");
  host.textContent = "";
  const present = S.result.per_class.filter((e) => e.pixels > 0);
  $("legend-count").textContent =
    present.length + (present.length === 1 ? " class" : " classes");

  for (const e of present) {
    const cls = S.classes[e.class_id];
    const li = document.createElement("li");
    li.className = "legend-item";
    li.dataset.cls = e.class_id;
    li.innerHTML = `<i class="legend-swatch" style="background:${cls.color}"></i>
      <span></span><b></b>`;
    li.querySelector("span").textContent = cls.name;
    li.querySelector("b").textContent = e.share.toFixed(1) + "%";
    li.addEventListener("click", () => toggleSolo(e.class_id));
    host.appendChild(li);
  }
  paintSolo();
}

function toggleSolo(cls) {
  const s = S.viewer.state;
  s.solo = s.solo === cls ? null : cls;
  S.viewer.draw();
  paintSolo();
}

function paintSolo() {
  const solo = S.viewer.state.solo;
  document.querySelectorAll(".legend-item").forEach((li) => {
    const c = Number(li.dataset.cls);
    li.classList.toggle("is-solo", solo === c);
    li.classList.toggle("is-muted", solo !== null && solo !== c);
  });
  document.querySelectorAll("#crop-table tbody tr").forEach((tr) => {
    tr.classList.toggle("is-solo", solo === Number(tr.dataset.cls));
  });
}

/* ───────────────────────────  right panel  ─────────────────────────── */

function kpi(label, value, unit, accent, barPct) {
  const d = document.createElement("div");
  d.className = "kpi" + (accent ? " accent" : "");
  d.innerHTML = `<label></label><b><span></span><small></small></b>`;
  d.querySelector("label").textContent = label;
  d.querySelector("span").textContent = value;
  d.querySelector("small").textContent = unit || "";
  if (barPct !== undefined && barPct !== null) {
    const bar = document.createElement("div");
    bar.className = "bar";
    bar.innerHTML = `<i style="width:${Math.max(0, Math.min(100, barPct))}%"></i>`;
    d.appendChild(bar);
  }
  return d;
}

function buildKpis() {
  const host = $("kpis");
  host.textContent = "";
  const o = S.result.overall;

  if (S.result.has_truth) {
    host.appendChild(kpi("Accuracy", fmt(o.accuracy, 1), "%", true, o.accuracy));
  } else {
    host.appendChild(kpi("Mapped area", fmt(o.labelled_area_ha, 0), "ha", true));
  }
  host.appendChild(kpi("Confidence", fmt(o.mean_confidence, 1), "%", false, o.mean_confidence));
  host.appendChild(kpi("Crops found", o.classes_found, ""));
}

function buildNotes() {
  const host = $("notes");
  host.textContent = "";
  const p = S.result;

  for (const w of p.warnings || []) {
    const d = document.createElement("div");
    d.className = "note";
    d.textContent = w;
    host.appendChild(d);
  }
  if (!p.has_truth) {
    const d = document.createElement("div");
    d.className = "note info";
    d.textContent =
      "No ground truth in this bundle, so accuracy and IoU cannot be measured — " +
      "only what the model thinks it sees.";
    host.appendChild(d);
  }
  const cloudCount = (p.cloudy || []).filter(Boolean).length;
  if (cloudCount) {
    const d = document.createElement("div");
    d.className = "note";
    d.textContent =
      `${cloudCount} of ${p.n_dates} dates look cloudy. The model still uses them; ` +
      "its attention usually drops on those days.";
    host.appendChild(d);
  }
}

function iouClass(v) {
  if (v === null || v === undefined) return "";
  return v >= 70 ? "iou-good" : v >= 40 ? "iou-mid" : "iou-bad";
}

function buildTable() {
  const tbody = $("crop-table").querySelector("tbody");
  tbody.textContent = "";
  const showIou = S.result.has_truth;
  document.querySelectorAll(".iou-col").forEach((th) => (th.hidden = !showIou));

  for (const e of S.result.per_class) {
    if (e.pixels === 0 && !e.truth_pixels) continue;
    const cls = S.classes[e.class_id];
    const tr = document.createElement("tr");
    tr.dataset.cls = e.class_id;
    tr.innerHTML = `
      <td><span class="cell-crop"><i style="background:${cls.color}"></i><span></span></span></td>
      <td class="num"></td><td class="num"></td><td class="num"></td>
      <td class="num"></td><td class="num iou-col ${iouClass(e.iou)}"></td>`;
    const tds = tr.querySelectorAll("td");
    tds[0].querySelector(".cell-crop span").textContent = cls.name;
    tds[0].querySelector(".cell-crop span").title = cls.name;
    tds[1].textContent = e.area_ha ? e.area_ha.toFixed(1) : "—";
    tds[2].textContent = e.share ? e.share.toFixed(1) + "%" : "—";
    tds[3].textContent = e.fields || "—";
    tds[4].textContent = e.confidence !== null ? e.confidence.toFixed(0) + "%" : "—";
    tds[5].textContent = e.iou !== null && e.iou !== undefined ? e.iou.toFixed(0) : "—";
    tds[5].hidden = !showIou;
    tr.addEventListener("click", () => toggleSolo(e.class_id));
    tbody.appendChild(tr);
  }
}

function buildCharts(frame) {
  const p = S.result;
  const heads = p.attention_time;
  const mean = heads[0].map((_, i) =>
    heads.reduce((sum, h) => sum + h[i], 0) / heads.length
  );
  attentionChart($("att-chart"), {
    dates: p.dates,
    weights: mean,
    frame,
    cloudy: p.cloudy,
    onPick: setFrame,
  });
  headsStrip($("att-heads"), { heads, frame });
}

async function showMap() {
  const p = S.result;
  const card = $("map-card");
  if (!p.bounds && !p.polygon) {
    card.hidden = true;
    return;
  }
  const ok = await S.map.show({
    bounds: p.bounds,
    polygon: p.polygon,
    overlayUrl: S.viewer.predictionDataUrl(1),
    opacity: S.viewer.state.opacity,
  });
  card.hidden = !ok;
  if (ok && p.bounds) {
    const [s, w, n, e] = p.bounds;
    $("coord-tag").textContent =
      `${((s + n) / 2).toFixed(3)}°N ${((w + e) / 2).toFixed(3)}°E`;
    $("map-hint").textContent =
      "The prediction laid back on the ground it came from. The footprint is " +
      "squared off to north, so it can sit a few metres from the true corners.";
  }
}

/* ───────────────────────────  time  ─────────────────────────── */

function buildTimeline() {
  const host = $("timeline");
  host.textContent = "";
  const p = S.result;
  const days = p.day_offsets;
  const lo = Math.min(...days);
  const hi = Math.max(...days);
  const span = Math.max(hi - lo, 1);

  let lastMonth = null;
  p.dates.forEach((iso, i) => {
    const pct = ((days[i] - lo) / span) * 100;
    const tick = document.createElement("div");
    tick.className = "tick" + (p.cloudy[i] ? " is-cloudy" : "");
    tick.style.left = pct + "%";
    tick.dataset.i = i;
    tick.title = iso + (p.cloudy[i] ? " · likely cloud" : "");
    tick.addEventListener("click", () => setFrame(i));
    host.appendChild(tick);

    const month = iso.slice(0, 7);
    if (month !== lastMonth) {
      lastMonth = month;
      const lab = document.createElement("div");
      lab.className = "month";
      lab.style.left = pct + "%";
      lab.textContent = new Date(iso).toLocaleString("en", { month: "short" });
      host.appendChild(lab);
    }
  });
}

/** Thumbnails of every date, drawn straight out of the sprite sheet. */
function buildFilmstrip() {
  const host = $("filmstrip");
  const p = S.result;
  host.textContent = "";
  $("strip-count").textContent = `${p.n_dates} dates`;
  $("filmstrip-card").hidden = false;

  const sheet = S.viewer.data.sheet;
  const row = S.viewer.state.base === "ndvi" ? 128 : 0;

  p.dates.forEach((iso, i) => {
    const wrap = document.createElement("div");
    wrap.className = "frame" + (p.cloudy[i] ? " is-cloudy" : "");
    wrap.dataset.i = i;
    const c = document.createElement("canvas");
    c.width = 128;
    c.height = 128;
    c.getContext("2d").drawImage(sheet, i * 128, row, 128, 128, 0, 0, 128, 128);
    const label = document.createElement("b");
    label.textContent = iso.slice(5).replace("-", "/");
    wrap.append(c, label);
    wrap.title = iso;
    wrap.addEventListener("click", () => setFrame(i));
    host.appendChild(wrap);
  });
}

function setFrame(i) {
  const p = S.result;
  if (!p) return;
  i = Math.max(0, Math.min(p.n_dates - 1, i));
  S.viewer.set("frame", i);
  $("date-slider").value = i;
  $("date-label").textContent = p.dates[i];
  $("date-index").textContent = `acquisition ${i + 1} of ${p.n_dates}`;
  $("cloud-flag").hidden = !p.cloudy[i];
  $("canvas-badge").textContent =
    (S.viewer.state.base === "ndvi" ? "greenness" : "true colour") + " · " + p.dates[i];
  document.querySelectorAll(".tick").forEach((t) => {
    t.classList.toggle("is-on", Number(t.dataset.i) === i);
  });
  document.querySelectorAll(".frame").forEach((f) => {
    const on = Number(f.dataset.i) === i;
    f.classList.toggle("is-on", on);
    // Nudge the strip sideways only. scrollIntoView would also scroll the whole
    // column, dragging the picture off screen every time the date changes.
    if (on) {
      const strip = f.parentElement;
      const fr = f.getBoundingClientRect();
      const sr = strip.getBoundingClientRect();
      if (fr.left < sr.left || fr.right > sr.right) {
        strip.scrollLeft += fr.left - sr.left - (sr.width - fr.width) / 2;
      }
    }
  });
  buildCharts(i);
  positionWipe();
}

function stopPlay() {
  if (S.playing) clearInterval(S.playing);
  S.playing = null;
  const btn = $("play-btn");
  btn.classList.remove("is-on");
  btn.title = "Play the year";
  btn.innerHTML = PLAY_ICON;
}

function togglePlay() {
  if (S.playing) return stopPlay();
  if (!S.result) return;

  // Sitting on the last date? Start over rather than doing nothing.
  if (Number($("date-slider").value) >= S.result.n_dates - 1) setFrame(0);

  const btn = $("play-btn");
  btn.classList.add("is-on");
  btn.title = "Pause";
  btn.innerHTML = PAUSE_ICON;

  S.playing = setInterval(() => {
    const next = Number($("date-slider").value) + 1;
    if (next >= S.result.n_dates) {
      // One pass through the year, then it stops on the final image.
      setFrame(S.result.n_dates - 1);
      stopPlay();
      return;
    }
    setFrame(next);
  }, 320);
}

/* ───────────────────────────  compare wipe  ─────────────────────────── */

function positionWipe() {
  const handle = $("wipe-handle");
  if (!S.result || !S.viewer.state.compare || !S.viewer.hasOverlay()) {
    handle.hidden = true;
    return;
  }
  handle.hidden = false;
  handle.style.left = S.viewer.state.wipe * 100 + "%";
}

function wireWipe() {
  const canvas = $("canvas");
  let dragging = false;

  const move = (ev) => {
    const rect = canvas.getBoundingClientRect();
    const x = (ev.clientX - rect.left) / rect.width;
    S.viewer.set("wipe", Math.max(0, Math.min(1, x)));
    positionWipe();
  };

  $("wipe-handle").addEventListener("pointerdown", (ev) => {
    dragging = true;
    ev.target.setPointerCapture?.(ev.pointerId);
    ev.preventDefault();
  });
  window.addEventListener("pointermove", (ev) => dragging && move(ev));
  window.addEventListener("pointerup", () => (dragging = false));
  window.addEventListener("resize", positionWipe);
}

/* ───────────────────────────  controls  ─────────────────────────── */

function wireControls() {
  $("date-slider").addEventListener("input", (e) => setFrame(Number(e.target.value)));
  $("play-btn").addEventListener("click", togglePlay);

  $("opacity").addEventListener("input", (e) => {
    const v = Number(e.target.value) / 100;
    $("opacity-val").textContent = e.target.value + "%";
    S.viewer.set("opacity", v);
    S.map.setOpacity(v);
  });

  document.querySelectorAll("#base-seg .seg-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("#base-seg .seg-btn").forEach((b) => {
        b.classList.remove("is-on");
        b.setAttribute("aria-checked", "false");
      });
      btn.classList.add("is-on");
      btn.setAttribute("aria-checked", "true");
      S.viewer.set("base", btn.dataset.base);
      if (S.result) {
        buildFilmstrip();
        setFrame(Number($("date-slider").value));
      }
    });
  });

  $("compare-btn").addEventListener("click", (e) => {
    const on = !S.viewer.state.compare;
    S.viewer.set("compare", on);
    e.currentTarget.classList.toggle("is-on", on);
    positionWipe();
  });

  $("grid-btn").addEventListener("click", (e) => {
    const on = !S.viewer.state.grid;
    S.viewer.set("grid", on);
    e.currentTarget.classList.toggle("is-on", on);
  });

  wireWipe();
  wireProbe();

  document.addEventListener("keydown", (ev) => {
    if (!S.result || ev.target.tagName === "INPUT") return;
    if (ev.key === "ArrowRight") setFrame(Number($("date-slider").value) + 1);
    if (ev.key === "ArrowLeft") setFrame(Number($("date-slider").value) - 1);
    if (ev.key === " ") { ev.preventDefault(); togglePlay(); }
  });
}

function wireProbe() {
  const canvas = $("canvas");
  const out = $("readout");

  canvas.addEventListener("mousemove", (ev) => {
    const hit = S.viewer.probe(ev.clientX, ev.clientY);
    if (!hit || hit.pred === null) { out.hidden = true; return; }
    const p = S.classes[hit.pred];
    const rows = [
      `<span class="swatch" style="background:${p.color}"></span>${p.name}`,
      `confidence ${(hit.conf * 100).toFixed(0)}%`,
    ];
    if (hit.truth !== null && hit.truth !== undefined) {
      const t = S.classes[hit.truth];
      const wrong = hit.truth !== hit.pred && hit.truth !== 19;
      rows.push(
        `<span class="${wrong ? "miss" : ""}">truth: ${t.name}</span>`
      );
    }
    rows.push(`x ${hit.metres[0]} m · y ${hit.metres[1]} m`);
    out.innerHTML = rows.join("<br>");
    out.hidden = false;
  });
  canvas.addEventListener("mouseleave", () => (out.hidden = true));
}

boot();
