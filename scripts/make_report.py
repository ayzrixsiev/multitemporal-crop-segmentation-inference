import argparse
import base64
import io
import json
import os
import pickle as pkl
from datetime import datetime

import numpy as np

PASTIS_CLASSES = [
    "Background",
    "Meadow",
    "Soft winter wheat",
    "Corn",
    "Winter barley",
    "Winter rapeseed",
    "Spring barley",
    "Sunflower",
    "Grapevine",
    "Beet",
    "Winter triticale",
    "Winter durum wheat",
    "Fruits, vegetables, flowers",
    "Potatoes",
    "Leguminous fodder",
    "Soybeans",
    "Orchard",
    "Mixed cereal",
    "Sorghum",
    "Void",
]

CLASS_COLORS = [
    "#f0efec",
    "#1baf7a",
    "#2a78d6",
    "#eda100",
    "#9085e9",
    "#e87ba4",
    "#86b6ef",
    "#eb6834",
    "#4a3aa7",
    "#d55181",
    "#008300",
    "#0d366b",
    "#e34948",
    "#c98500",
    "#199e70",
    "#3987e5",
    "#184f95",
    "#d95926",
    "#6da7ec",
    "#ffffff",
]

SEQ_BLUE = [
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
]


def load_run(res_dir, fold):
    fold_dir = os.path.join(res_dir, "Fold_{}".format(fold))
    if not os.path.isdir(fold_dir):
        raise SystemExit(
            "No such fold directory: {}\n"
            "Point --res_dir at the folder train_semantic.py wrote to.".format(fold_dir)
        )

    with open(os.path.join(fold_dir, "trainlog.json")) as f:
        raw_log = json.load(f)
    trainlog = {int(k): v for k, v in raw_log.items()}

    test_metrics = None
    test_path = os.path.join(fold_dir, "test_metrics.json")
    if os.path.exists(test_path):
        with open(test_path) as f:
            test_metrics = json.load(f)

    conf_mat = None
    cm_path = os.path.join(fold_dir, "conf_mat.pkl")
    if os.path.exists(cm_path):
        with open(cm_path, "rb") as f:
            conf_mat = np.asarray(pkl.load(f), dtype=np.float64)

    config = {}
    conf_path = os.path.join(res_dir, "conf.json")
    if os.path.exists(conf_path):
        with open(conf_path) as f:
            config = json.load(f)

    return trainlog, test_metrics, conf_mat, config


def per_class_metrics(conf_mat, ignore_index=-1):
    """IoU / precision / recall / F1 / support per class, void dropped."""
    cm = conf_mat.copy()
    keep = [i for i in range(cm.shape[0]) if i != (ignore_index % cm.shape[0])]
    cm = cm[np.ix_(keep, keep)]

    rows = []
    with np.errstate(divide="ignore", invalid="ignore"):
        for j, cls in enumerate(keep):
            tp = cm[j, j]
            fp = cm[:, j].sum() - tp
            fn = cm[j, :].sum() - tp
            rows.append(
                {
                    "index": cls,
                    "name": PASTIS_CLASSES[cls],
                    "iou": (
                        float(tp / (tp + fp + fn)) if (tp + fp + fn) else float("nan")
                    ),
                    "precision": float(tp / (tp + fp)) if (tp + fp) else float("nan"),
                    "recall": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
                    "f1": (
                        float(2 * tp / (2 * tp + fp + fn))
                        if (tp + fp + fn)
                        else float("nan")
                    ),
                    "support": float(cm[j, :].sum()),
                }
            )
    return rows, cm


def _png_data_uri(fig):
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _colorize(label_map):
    """(H, W) int labels -> (H, W, 3) uint8 using CLASS_COLORS."""
    rgb = np.zeros(label_map.shape + (3,), dtype=np.uint8)
    for cls, hexcol in enumerate(CLASS_COLORS):
        h = hexcol.lstrip("#")
        rgb[label_map == cls] = [int(h[i : i + 2], 16) for i in (0, 2, 4)]
    return rgb


def qualitative_panels(dataset_folder, weights, fold, config, n_samples, device):
    """Run the trained model on a few test patches and return image panels."""
    import matplotlib.pyplot as plt
    import torch

    from src import model_utils
    from src.dataset import Pastis_Dataset

    fold_sequence = [
        [[1, 2, 3], [4], [5]],
        [[2, 3, 4], [5], [1]],
        [[3, 4, 5], [1], [2]],
        [[4, 5, 1], [2], [3]],
        [[5, 1, 2], [3], [4]],
    ]
    test_fold = fold_sequence[fold - 1][2]

    dt = Pastis_Dataset(
        folder=dataset_folder,
        norm=True,
        reference_date=config.get("ref_date", "2018-09-01"),
        mono_date=None,
        target="semantic",
        sats=["S2"],
        folds=test_fold,
    )

    cfg = argparse.Namespace(**config)
    model = model_utils.get_model(cfg, mode="semantic")
    state = torch.load(weights, map_location="cpu")
    model.load_state_dict(state["state_dict"])
    model = model.to(device).eval()

    scored = []
    for i in range(min(len(dt), 60)):
        _, y = dt[i]
        n_cls = len(np.unique(y.numpy()))
        scored.append((n_cls, i))
    scored.sort(reverse=True)
    picks = [i for _, i in scored[:n_samples]]

    mean, std = dt.norm["S2"]
    panels = []
    attention = None

    for rank, idx in enumerate(picks):
        (x, dates), y = dt[idx]
        xb = x.unsqueeze(0).to(device)
        db = dates.unsqueeze(0).to(device)

        with torch.no_grad():
            out, att = model(xb, batch_positions=db, return_att=True)
        pred = out.argmax(dim=1)[0].cpu().numpy()
        truth = y.numpy()

        raw = x * std[None, :, None, None] + mean[None, :, None, None]
        frame = int(raw.mean(dim=(1, 2, 3)).argmin())
        rgb = raw[frame, [2, 1, 0]].permute(1, 2, 0).numpy()
        lo, hi = np.percentile(rgb, [2, 98])
        rgb = np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1)

        images = []
        for arr, caption in (
            (rgb, "Sentinel-2 true colour"),
            (_colorize(truth), "Ground truth"),
            (_colorize(pred), "U-TAE prediction"),
        ):
            fig, ax = plt.subplots(figsize=(2.6, 2.6))
            ax.imshow(arr, interpolation="nearest")
            ax.set_axis_off()
            images.append({"src": _png_data_uri(fig), "caption": caption})

        present = sorted(set(np.unique(truth)) | set(np.unique(pred)))
        agree = (
            float((pred == truth)[truth != 19].mean() * 100)
            if (truth != 19).any()
            else float("nan")
        )

        panels.append(
            {
                "images": images,
                "legend": [
                    {"name": PASTIS_CLASSES[c], "color": CLASS_COLORS[c]}
                    for c in present
                    if c != 19
                ],
                "agreement": agree,
                "n_dates": int(x.shape[0]),
            }
        )

        if rank == 0:
            attention = attention_figure(att, dates, config)

    return panels, attention


def attention_figure(att, dates, config):
    import matplotlib.pyplot as plt

    a = att[:, 0].mean(dim=(-1, -2)).cpu().numpy()  # (n_head, T)
    days = dates.cpu().numpy()
    ref = datetime(*map(int, config.get("ref_date", "2018-09-01").split("-")))
    labels = [(np.datetime64(ref.date()) + np.timedelta64(int(d), "D")) for d in days]

    muted = "#898781"
    fig, ax = plt.subplots(figsize=(9.5, 2.9))
    im = ax.imshow(a, aspect="auto", cmap="Blues", interpolation="nearest")

    step = max(1, len(labels) // 12)
    ax.set_xticks(range(0, len(labels), step))
    ax.set_xticklabels(
        [str(labels[i])[:7] for i in range(0, len(labels), step)],
        rotation=45,
        ha="right",
        fontsize=8,
        color=muted,
    )
    ax.set_yticks([0, a.shape[0] - 1])
    ax.set_yticklabels(
        ["head 1", "head {}".format(a.shape[0])], fontsize=8, color=muted
    )
    ax.set_xlabel("acquisition date", fontsize=9, color=muted)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    cb = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.025)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, length=0, colors=muted)
    cb.set_label("attention weight", fontsize=8, color=muted)

    return _png_data_uri(fig)


def svg_line_chart(series, y_label, chart_id, y_pct=False):
    W, H = 460, 230
    pad = {"t": 14, "r": 16, "b": 34, "l": 46}
    pw, ph = W - pad["l"] - pad["r"], H - pad["t"] - pad["b"]

    xs = [p[0] for s in series for p in s["points"]]
    ys = [p[1] for s in series for p in s["points"]]
    if not xs:
        return ""
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    span = (y1 - y0) or 1.0
    y1 = y1 + span * 0.08
    # never extend the axis below zero for a quantity that cannot be negative
    y0 = max(0.0, y0 - span * 0.08) if min(ys) >= 0 else y0 - span * 0.08

    def px(x):
        return pad["l"] + (0 if x1 == x0 else (x - x0) / (x1 - x0)) * pw

    def py(y):
        return pad["t"] + (1 - (y - y0) / (y1 - y0)) * ph

    parts = ['<svg viewBox="0 0 {} {}" class="chart" role="img">'.format(W, H)]

    ticks = [y0 + (y1 - y0) * i / 4 for i in range(5)]
    for t in ticks:
        parts.append(
            '<line x1="{:.1f}" y1="{:.1f}" x2="{:.1f}" y2="{:.1f}" class="grid"/>'.format(
                pad["l"], py(t), pad["l"] + pw, py(t)
            )
        )
        parts.append(
            '<text x="{:.1f}" y="{:.1f}" class="tick tick-y">{}</text>'.format(
                pad["l"] - 8, py(t) + 3.5, _fmt(t, y_pct)
            )
        )

    xt = sorted({x0, (x0 + x1) // 2, x1})
    for t in xt:
        parts.append(
            '<text x="{:.1f}" y="{:.1f}" class="tick tick-x">{}</text>'.format(
                px(t), pad["t"] + ph + 18, int(t)
            )
        )
    parts.append(
        '<line x1="{:.1f}" y1="{:.1f}" x2="{:.1f}" y2="{:.1f}" class="axis"/>'.format(
            pad["l"], pad["t"] + ph, pad["l"] + pw, pad["t"] + ph
        )
    )

    end_ys = [py(s["points"][-1][1]) for s in series]
    labels_fit = (
        len(end_ys) < 2
        or min(abs(a - b) for i, a in enumerate(end_ys) for b in end_ys[i + 1 :]) >= 14
    )

    for s in series:
        pts = s["points"]
        d = " ".join(
            "{}{:.1f},{:.1f}".format("M" if i == 0 else "L", px(x), py(y))
            for i, (x, y) in enumerate(pts)
        )
        parts.append(
            '<path d="{}" fill="none" stroke="var(--{})" stroke-width="2" '
            'stroke-linejoin="round" stroke-linecap="round"/>'.format(d, s["slot"])
        )
        ex, ey = pts[-1]
        parts.append(
            '<circle cx="{:.1f}" cy="{:.1f}" r="4" fill="var(--{})" '
            'stroke="var(--surface-1)" stroke-width="2"/>'.format(
                px(ex), py(ey), s["slot"]
            )
        )
        if labels_fit:
            parts.append(
                '<text x="{:.1f}" y="{:.1f}" class="endlabel" text-anchor="end">{}</text>'.format(
                    px(ex) - 8, py(ey) - 9, _fmt(ey, y_pct)
                )
            )

    step = pw / max(len(series[0]["points"]) - 1, 1)
    for i, (x, _) in enumerate(series[0]["points"]):
        vals = []
        for s in series:
            match = [v for e, v in s["points"] if e == x]
            if match:
                vals.append("{}: {}".format(s["label"], _fmt(match[0], y_pct)))
        parts.append(
            '<rect x="{:.1f}" y="{:.1f}" width="{:.1f}" height="{:.1f}" fill="transparent" '
            'class="hit" data-tip="Epoch {} &#8212; {}"/>'.format(
                px(x) - step / 2,
                pad["t"],
                max(step, 6),
                ph,
                int(x),
                " &#183; ".join(vals),
            )
        )

    parts.append("</svg>")

    legend = "".join(
        '<span class="key"><i style="background:var(--{})"></i>{}</span>'.format(
            s["slot"], s["label"]
        )
        for s in series
    )
    return (
        '<figure class="card"><figcaption><h3>{}</h3><div class="legend">{}</div>'
        "</figcaption>{}</figure>".format(y_label, legend, "".join(parts))
    )


def svg_bar_chart(rows):
    rows = [r for r in rows if not np.isnan(r["iou"])]
    rows = sorted(rows, key=lambda r: r["iou"], reverse=True)

    row_h, gap = 21, 6
    W = 620
    label_w, value_w = 190, 46
    track = W - label_w - value_w
    H = len(rows) * (row_h + gap)

    parts = ['<svg viewBox="0 0 {} {}" class="chart" role="img">'.format(W, H)]
    for i, r in enumerate(rows):
        y = i * (row_h + gap)
        bar_h = min(row_h, 24)
        w = max(track * r["iou"], 1.5)
        parts.append(
            '<text x="{}" y="{:.1f}" class="barlabel" text-anchor="end">{}</text>'.format(
                label_w - 12, y + bar_h * 0.72, _esc(r["name"])
            )
        )
        parts.append(
            '<path d="{}" fill="var(--series-1)" class="hit" data-tip="{} &#8212; '
            'IoU {:.3f} &#183; precision {:.3f} &#183; recall {:.3f}"/>'.format(
                _bar_path(label_w, y, w, bar_h),
                _esc(r["name"]),
                r["iou"],
                r["precision"],
                r["recall"],
            )
        )
        parts.append(
            '<text x="{:.1f}" y="{:.1f}" class="barvalue">{:.3f}</text>'.format(
                label_w + w + 9, y + bar_h * 0.72, r["iou"]
            )
        )
    parts.append("</svg>")
    return "".join(parts)


def _bar_path(x, y, w, h, r=4):
    """Horizontal bar: square where it meets the baseline, rounded at the data end."""
    r = min(r, w, h / 2)
    return (
        "M{x:.1f},{y:.1f} H{a:.1f} A{r},{r} 0 0 1 {b:.1f},{c:.1f} V{d:.1f} "
        "A{r},{r} 0 0 1 {a:.1f},{e:.1f} H{x:.1f} Z"
    ).format(x=x, y=y, a=x + w - r, b=x + w, c=y + r, d=y + h - r, e=y + h, r=r)


def html_confusion(cm, rows):
    with np.errstate(divide="ignore", invalid="ignore"):
        norm = cm / cm.sum(axis=1, keepdims=True)
    norm = np.nan_to_num(norm)

    names = [r["name"] for r in rows]
    n = len(names)

    head = "".join(
        '<th class="cm-col"><span>{}</span></th>'.format(_esc(nm)) for nm in names
    )
    body = []
    for i in range(n):
        cells = []
        for j in range(n):
            v = norm[i, j]
            step = (
                SEQ_BLUE[min(int(v * len(SEQ_BLUE)), len(SEQ_BLUE) - 1)]
                if v > 0
                else "transparent"
            )
            cells.append(
                '<td class="cm-cell hit" style="background:{}" '
                'data-tip="truth {} &#8594; predicted {}: {:.1%} ({:,.0f} px)"></td>'.format(
                    step, _esc(names[i]), _esc(names[j]), v, cm[i, j]
                )
            )
        body.append(
            '<tr><th class="cm-row">{}</th>{}</tr>'.format(
                _esc(names[i]), "".join(cells)
            )
        )

    scale = "".join('<i style="background:{}"></i>'.format(c) for c in SEQ_BLUE)
    return (
        '<div class="cm-scroll"><table class="cm">'
        '<thead><tr><th class="cm-corner"></th>{}</tr></thead>'
        "<tbody>{}</tbody></table></div>"
        '<div class="cm-key"><span>0%</span><div class="ramp">{}</div><span>100% of a class’s pixels</span></div>'.format(
            head, "".join(body), scale
        )
    )


def html_table(rows):
    body = "".join(
        "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{:,.0f}</td></tr>".format(
            _esc(r["name"]),
            _fmt(r["iou"], False, 3),
            _fmt(r["precision"], False, 3),
            _fmt(r["recall"], False, 3),
            _fmt(r["f1"], False, 3),
            r["support"],
        )
        for r in sorted(rows, key=lambda r: (np.isnan(r["iou"]), -r["iou"]))
    )
    return (
        '<table class="data"><thead><tr><th>Class</th><th>IoU</th><th>Precision</th>'
        "<th>Recall</th><th>F1</th><th>Test pixels</th></tr></thead>"
        "<tbody>{}</tbody></table>".format(body)
    )


def _fmt(v, pct=False, nd=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return "{:.1f}".format(v) if pct else "{:.{}f}".format(v, nd)


def _esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# --------------------------------------------------------------------------- #
# page
# --------------------------------------------------------------------------- #

CSS = """
*,*::before,*::after{box-sizing:border-box}
:root{
  color-scheme:light;
  --page:#f9f9f7; --surface-1:#fcfcfb;
  --text-primary:#0b0b0b; --text-secondary:#52514e; --text-muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,0.10);
  --series-1:#2a78d6; --series-2:#eb6834;
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    color-scheme:dark;
    --page:#0d0d0d; --surface-1:#1a1a19;
    --text-primary:#ffffff; --text-secondary:#c3c2b7; --text-muted:#898781;
    --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,0.10);
    --series-1:#3987e5; --series-2:#d95926;
  }
}
:root[data-theme="dark"]{
  color-scheme:dark;
  --page:#0d0d0d; --surface-1:#1a1a19;
  --text-primary:#ffffff; --text-secondary:#c3c2b7; --text-muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,0.10);
  --series-1:#3987e5; --series-2:#d95926;
}
body{
  margin:0; background:var(--page); color:var(--text-primary);
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif;
  font-size:15px; line-height:1.55;
}
.wrap{max-width:1080px; margin:0 auto; padding:56px 28px 96px}
header{margin-bottom:44px}
h1{font-size:29px; font-weight:600; letter-spacing:-0.02em; margin:0 0 6px}
.sub{color:var(--text-secondary); margin:0; font-size:15px}
h2{font-size:13px; font-weight:600; letter-spacing:0.08em; text-transform:uppercase;
   color:var(--text-muted); margin:56px 0 18px}
h3{font-size:14px; font-weight:600; margin:0; color:var(--text-primary)}
p.note{color:var(--text-secondary); margin:0 0 20px; max-width:62ch; font-size:14px}

.hero{
  background:var(--surface-1); border:1px solid var(--border); border-radius:14px;
  padding:30px 32px; display:flex; flex-wrap:wrap; gap:36px; align-items:flex-end;
}
.hero .figure{font-size:58px; font-weight:600; line-height:1; letter-spacing:-0.03em}
.hero .figure small{font-size:20px; font-weight:500; color:var(--text-secondary); margin-left:8px}
.hero .cap{color:var(--text-secondary); font-size:14px; margin-top:8px}
.kpis{display:flex; flex-wrap:wrap; gap:34px; margin-left:auto}
.kpi .v{font-size:23px; font-weight:600; letter-spacing:-0.01em}
.kpi .l{font-size:12.5px; color:var(--text-muted)}

.grid2{display:grid; grid-template-columns:repeat(auto-fit,minmax(330px,1fr)); gap:18px}
.card{
  background:var(--surface-1); border:1px solid var(--border); border-radius:14px;
  padding:18px 20px 12px; margin:0;
}
.card figcaption{display:flex; justify-content:space-between; align-items:baseline;
                 gap:14px; margin-bottom:6px; flex-wrap:wrap}
.chart{width:100%; height:auto; display:block; overflow:visible}
.grid{stroke:var(--grid); stroke-width:1}
.axis{stroke:var(--axis); stroke-width:1}
.tick{fill:var(--text-muted); font-size:10px; font-variant-numeric:tabular-nums}
.tick-y{text-anchor:end}
.tick-x{text-anchor:middle}
.endlabel{fill:var(--text-secondary); font-size:11px; font-weight:600}
.barlabel{fill:var(--text-secondary); font-size:12px}
.barvalue{fill:var(--text-primary); font-size:12px; font-weight:600;
          font-variant-numeric:tabular-nums}
.legend{display:flex; gap:14px}
.key{display:inline-flex; align-items:center; gap:6px; font-size:12.5px;
     color:var(--text-secondary)}
.key i{width:11px; height:11px; border-radius:3px; display:inline-block}
.hit{cursor:crosshair}

.cm-scroll{overflow-x:auto; background:var(--surface-1); border:1px solid var(--border);
           border-radius:14px; padding:16px}
table.cm{border-collapse:separate; border-spacing:2px; font-size:11px}
.cm-cell{width:19px; height:19px; border-radius:3px; padding:0;
         background-clip:padding-box}
.cm-row{font-weight:400; color:var(--text-secondary); text-align:right;
        padding-right:10px; white-space:nowrap; max-width:180px}
.cm-col{height:118px; vertical-align:bottom; padding:0 0 6px}
.cm-col span{writing-mode:vertical-rl; transform:rotate(180deg);
             color:var(--text-secondary); font-weight:400; white-space:nowrap}
.cm-key{display:flex; align-items:center; gap:10px; margin-top:12px;
        color:var(--text-muted); font-size:12px}
.cm-key .ramp{display:flex}
.cm-key .ramp i{width:17px; height:9px}

table.data{width:100%; border-collapse:collapse; font-size:13.5px; margin-top:8px}
table.data th{text-align:right; font-weight:600; color:var(--text-muted);
              font-size:12px; padding:8px 12px; border-bottom:1px solid var(--border)}
table.data th:first-child{text-align:left}
table.data td{text-align:right; padding:7px 12px; border-bottom:1px solid var(--border);
              font-variant-numeric:tabular-nums; color:var(--text-secondary)}
table.data td:first-child{text-align:left; color:var(--text-primary)}

.panel{background:var(--surface-1); border:1px solid var(--border); border-radius:14px;
       padding:18px 20px; margin-bottom:18px}
.panel .imgs{display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:16px}
.panel figure{margin:0}
.panel img{width:100%; height:auto; border-radius:8px; display:block; image-rendering:pixelated}
.panel figcaption{font-size:12.5px; color:var(--text-muted); margin-top:7px}
.panel .meta{font-size:13px; color:var(--text-secondary); margin-bottom:14px}
.swatches{display:flex; flex-wrap:wrap; gap:8px 16px; margin-top:14px}
.swatches .key i{border:1px solid var(--border)}
.wide img{width:100%; height:auto; display:block}

#tip{
  position:fixed; pointer-events:none; opacity:0; transition:opacity .1s;
  background:var(--text-primary); color:var(--surface-1); font-size:12.5px;
  padding:6px 10px; border-radius:7px; max-width:320px; z-index:9; line-height:1.4;
}
footer{margin-top:64px; padding-top:22px; border-top:1px solid var(--border);
       color:var(--text-muted); font-size:12.5px}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12.5px;
     background:var(--surface-1); border:1px solid var(--border); border-radius:6px;
     padding:2px 6px}
"""

JS = """
(function(){
  var tip=document.createElement('div'); tip.id='tip'; document.body.appendChild(tip);
  function show(e){
    var t=e.target.closest('[data-tip]'); if(!t){return hide();}
    tip.innerHTML=t.getAttribute('data-tip'); tip.style.opacity='1';
    var r=tip.getBoundingClientRect();
    var x=Math.min(e.clientX+14, window.innerWidth-r.width-10);
    var y=Math.max(e.clientY-r.height-12, 8);
    tip.style.left=x+'px'; tip.style.top=y+'px';
  }
  function hide(){tip.style.opacity='0';}
  document.addEventListener('mousemove',show);
  document.addEventListener('mouseleave',hide);
})();
"""


def build_html(trainlog, test_metrics, conf_mat, config, panels, attention, fold):
    epochs = sorted(trainlog)
    train_loss = [
        (e, trainlog[e]["train_loss"]) for e in epochs if "train_loss" in trainlog[e]
    ]
    val_loss = [
        (e, trainlog[e]["val_loss"]) for e in epochs if "val_loss" in trainlog[e]
    ]
    train_iou = [
        (e, trainlog[e]["train_IoU"]) for e in epochs if "train_IoU" in trainlog[e]
    ]
    val_iou = [(e, trainlog[e]["val_IoU"]) for e in epochs if "val_IoU" in trainlog[e]]

    best_val = max((v for _, v in val_iou), default=float("nan"))
    epoch_time = (
        np.mean(
            [
                trainlog[e]["train_epoch_time"]
                for e in epochs
                if "train_epoch_time" in trainlog[e]
            ]
        )
        if epochs
        else float("nan")
    )

    rows, cm = per_class_metrics(conf_mat) if conf_mat is not None else ([], None)

    test_iou = test_metrics["test_IoU"] if test_metrics else float("nan")
    test_acc = test_metrics["test_accuracy"] if test_metrics else float("nan")

    charts = []
    if train_loss:
        charts.append(
            svg_line_chart(
                (
                    [
                        {"points": train_loss, "slot": "series-1", "label": "Train"},
                        {"points": val_loss, "slot": "series-2", "label": "Validation"},
                    ]
                    if val_loss
                    else [{"points": train_loss, "slot": "series-1", "label": "Train"}]
                ),
                "Cross-entropy loss",
                "loss",
            )
        )
    if train_iou:
        charts.append(
            svg_line_chart(
                (
                    [
                        {"points": train_iou, "slot": "series-1", "label": "Train"},
                        {"points": val_iou, "slot": "series-2", "label": "Validation"},
                    ]
                    if val_iou
                    else [{"points": train_iou, "slot": "series-1", "label": "Train"}]
                ),
                "mIoU (%)",
                "miou",
                y_pct=True,
            )
        )

    panel_html = []
    for p in panels:
        imgs = "".join(
            '<figure><img src="{}" alt="{}"><figcaption>{}</figcaption></figure>'.format(
                im["src"], im["caption"], im["caption"]
            )
            for im in p["images"]
        )
        legend = "".join(
            '<span class="key"><i style="background:{}"></i>{}</span>'.format(
                c["color"], _esc(c["name"])
            )
            for c in p["legend"]
        )
        panel_html.append(
            '<div class="panel"><div class="meta">{} acquisitions over the year '
            "&#183; {:.1f}% of labelled pixels correct</div>"
            '<div class="imgs">{}</div><div class="swatches">{}</div></div>'.format(
                p["n_dates"], p["agreement"], imgs, legend
            )
        )

    sections = []
    if charts:
        sections.append(
            "<h2>Training dynamics</h2>"
            '<p class="note">Loss and mIoU are plotted separately &#8212; they share an '
            "x-axis but not a scale, and forcing them onto one plot would invent a "
            "relationship the data does not contain.</p>"
            '<div class="grid2">{}</div>'.format("".join(charts))
        )

    if rows:
        sections.append(
            "<h2>Per-class performance</h2>"
            '<p class="note">Intersection over union for each of the 19 scored classes on the '
            "held-out test fold. The void class is excluded, matching the loss weighting used "
            "during training.</p>"
            '<div class="card">{}</div>'.format(svg_bar_chart(rows))
        )
        sections.append(
            "<h2>Where the errors go</h2>"
            '<p class="note">Rows are ground truth, columns are the prediction, each row '
            "normalised to its own pixel count. A clean diagonal is the goal; bright "
            "off-diagonal cells name the crop pairs the model actually confuses.</p>"
            "{}".format(html_confusion(cm, rows))
        )

    if panel_html:
        sections.append(
            "<h2>Qualitative results</h2>"
            '<p class="note">Patches from the test fold the model never saw during training. '
            "The true-colour frame is the clearest single acquisition; the model saw the whole "
            "year.</p>{}".format("".join(panel_html))
        )

    if attention:
        sections.append(
            "<h2>What the temporal encoder looks at</h2>"
            '<p class="note">Each row is one L-TAE attention head, each column one Sentinel-2 '
            "acquisition, averaged over the patch. This is the part that makes U-TAE more than "
            "a U-Net: the heads learn to concentrate on the dates that separate crops and to "
            "walk away from clouded ones.</p>"
            '<div class="card wide"><img src="{}" alt="Temporal attention by head and date"></div>'.format(
                attention
            )
        )

    if rows:
        sections.append(
            "<h2>Full metrics</h2>"
            '<p class="note">Every number in the charts above, as text.</p>{}'.format(
                html_table(rows)
            )
        )

    cmd = (
        "uv run python train_semantic.py --dataset_folder &lt;PASTIS&gt; --fold {} "
        "--epochs {} --batch_size {} --lr {}".format(
            fold,
            config.get("epochs", "?"),
            config.get("batch_size", "?"),
            config.get("lr", "?"),
        )
    )

    return """<title>U-TAE on PASTIS</title>
<style>{css}</style>
<div class="wrap">
<header>
  <h1>Crop-type segmentation from Sentinel-2 time series</h1>
  <p class="sub">U-TAE trained on PASTIS, fold {fold} &#183; report generated {now}</p>
</header>

<div class="hero">
  <div>
    <div class="figure">{miou}<small>mIoU</small></div>
    <div class="cap">Mean intersection over union on the held-out test fold</div>
  </div>
  <div class="kpis">
    <div class="kpi"><div class="v">{acc}%</div><div class="l">Overall accuracy</div></div>
    <div class="kpi"><div class="v">{params}</div><div class="l">Parameters</div></div>
    <div class="kpi"><div class="v">{n_epochs}</div><div class="l">Epochs</div></div>
    <div class="kpi"><div class="v">{best_val}</div><div class="l">Best val mIoU</div></div>
    <div class="kpi"><div class="v">{etime}</div><div class="l">Per epoch</div></div>
  </div>
</div>

{sections}

<footer>
  Reproduce with <code>{cmd}</code>. Model: U-TAE (Sainte Fare Garnot &amp; Landrieu, ICCV 2021),
  reimplemented from VSainteuf/utae-paps.
</footer>
</div>
<script>{js}</script>
""".format(
        css=CSS,
        js=JS,
        fold=fold,
        now=datetime.now().strftime("%d %B %Y"),
        miou=_fmt(test_iou, True),
        acc=_fmt(test_acc, True),
        params="{:,}".format(config["N_params"]) if "N_params" in config else "—",
        n_epochs=len(epochs),
        best_val=_fmt(best_val, True),
        etime="{:.0f}s".format(epoch_time) if epoch_time == epoch_time else "—",
        sections="".join(sections),
        cmd=cmd,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--res_dir", default="./results")
    p.add_argument("--fold", default=1, type=int)
    p.add_argument("--out", default="./report/index.html")
    p.add_argument(
        "--dataset_folder",
        default=None,
        help="PASTIS root. Needed for the qualitative and attention panels.",
    )
    p.add_argument(
        "--weights",
        default=None,
        help="Checkpoint. Defaults to <res_dir>/Fold_<n>/model.pth.tar",
    )
    p.add_argument("--n_samples", default=3, type=int)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    trainlog, test_metrics, conf_mat, config = load_run(args.res_dir, args.fold)

    panels, attention = [], None
    if args.dataset_folder:
        weights = args.weights or os.path.join(
            args.res_dir, "Fold_{}".format(args.fold), "model.pth.tar"
        )
        if not os.path.exists(weights):
            raise SystemExit("No checkpoint at {}".format(weights))
        panels, attention = qualitative_panels(
            args.dataset_folder, weights, args.fold, config, args.n_samples, args.device
        )
    else:
        print("No --dataset_folder given: skipping qualitative and attention panels.")

    html = build_html(
        trainlog, test_metrics, conf_mat, config, panels, attention, args.fold
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    print("Wrote {} ({:.1f} KB)".format(args.out, os.path.getsize(args.out) / 1024))


if __name__ == "__main__":
    main()
