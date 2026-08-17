/* Canvas compositing.
 *
 * Everything the model produced is a 128x128 grid of numbers. This file turns
 * those grids into one picture: the satellite image underneath, the chosen
 * view painted on top, blended at whatever opacity the slider says.
 *
 * The work happens on a hidden 128x128 canvas and is then blown up to screen
 * size with smoothing switched off, so every pixel stays a crisp square instead
 * of being blurred into its neighbours.
 */

const SIZE = 128;
const VOID = 19; // unlabelled pixels; excluded from every score

export function hexToRgb(hex) {
  const h = hex.replace("#", "");
  return [
    parseInt(h.slice(0, 2), 16),
    parseInt(h.slice(2, 4), 16),
    parseInt(h.slice(4, 6), 16),
  ];
}

export function loadImage(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("could not load " + url));
    img.src = url;
  });
}

/** Pull the grey value out of every pixel of a single-channel PNG. */
export function toGrayArray(img, w = SIZE, h = SIZE) {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(img, 0, 0);
  const src = ctx.getImageData(0, 0, w, h).data;
  const out = new Uint8Array(w * h);
  for (let i = 0; i < out.length; i++) out[i] = src[i * 4];
  return out;
}

function scratch() {
  const c = document.createElement("canvas");
  c.width = SIZE;
  c.height = SIZE;
  return { canvas: c, ctx: c.getContext("2d", { willReadFrequently: true }) };
}

export class Viewer {
  constructor(canvas, palette) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.palette = palette.map((c) => hexToRgb(c.color));
    this.classes = palette;

    this.plain = scratch(); // base image only
    this.mixed = scratch(); // base image + overlays

    // Prediction, ground truth and mistakes are three ways of looking at the
    // same patch, so only one can be on at a time -- stacking them painted the
    // red error blobs over the ground truth and made both unreadable.
    // Uncertainty is separate: it is a glow laid over whichever view is chosen.
    this.state = {
      base: "rgb",
      frame: 0,
      opacity: 0.8,
      solo: null,
      compare: true,
      grid: false,
      wipe: 0.5,
      mode: "pred", // pred | truth | error | none
      uncertainty: false,
    };
    this.data = null;
  }

  /** Hand over one prediction: the sprite sheet plus every 128x128 map. */
  setData(data) {
    this.data = data;
    this.state.frame = 0;
    this.state.solo = null;
  }

  set(key, value) {
    this.state[key] = value;
    this.draw();
  }

  /** Is anything at all being drawn over the satellite image? */
  hasOverlay() {
    return this.state.mode !== "none" || this.state.uncertainty;
  }

  /** Paint the chosen date of the chosen base image into a scratch canvas. */
  _drawBase(target) {
    const { sheet } = this.data;
    const row = this.state.base === "ndvi" ? SIZE : 0;
    target.ctx.clearRect(0, 0, SIZE, SIZE);
    target.ctx.drawImage(
      sheet,
      this.state.frame * SIZE, row, SIZE, SIZE,
      0, 0, SIZE, SIZE
    );
  }

  _composite() {
    const s = this.state;
    const d = this.data;
    const img = this.mixed.ctx.getImageData(0, 0, SIZE, SIZE);
    const px = img.data;
    const alpha = s.opacity;

    const labels = s.mode === "pred" ? d.pred : s.mode === "truth" ? d.truth : null;
    const errorView = s.mode === "error" && d.error && d.pred && d.truth;

    for (let i = 0; i < SIZE * SIZE; i++) {
      const o = i * 4;
      let r = px[o], g = px[o + 1], b = px[o + 2];

      if (labels) {
        const cls = labels[i];
        const visible = s.solo === null || s.solo === cls;
        if (visible && cls < this.palette.length) {
          const [cr, cg, cb] = this.palette[cls];
          r = r + (cr - r) * alpha;
          g = g + (cg - g) * alpha;
          b = b + (cb - b) * alpha;
        }
      } else if (errorView) {
        // Wrong pixels in hard red; right ones keep a ghost of their crop colour
        // so the field shapes stay readable instead of a red blob on a photo.
        if (d.truth[i] === VOID) {
          const a = alpha * 0.35;
          r = r + (150 - r) * a; g = g + (150 - g) * a; b = b + (150 - b) * a;
        } else if (d.error[i] > 127) {
          r = r + (226 - r) * alpha;
          g = g + (42 - g) * alpha;
          b = b + (32 - b) * alpha;
        } else {
          const [cr, cg, cb] = this.palette[d.pred[i]] || [0, 0, 0];
          const a = alpha * 0.22;
          r = r + (cr - r) * a; g = g + (cg - g) * a; b = b + (cb - b) * a;
        }
      }

      // Uncertainty goes on last. Underneath the crop colours it was being
      // painted over at 80% and was all but invisible.
      if (s.uncertainty && d.conf) {
        const doubt = 1 - d.conf[i] / 255;
        if (doubt > 0.06) {
          const a = Math.min(1, (doubt - 0.06) / 0.45) * (0.4 + 0.6 * alpha);
          r = r + (255 - r) * a;
          g = g + (0 - g) * a;
          b = b + (110 - b) * a;
        }
      }

      px[o] = r; px[o + 1] = g; px[o + 2] = b;
    }
    this.mixed.ctx.putImageData(img, 0, 0);
  }

  draw() {
    const ctx = this.ctx;
    const W = this.canvas.width;
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, W, W);
    if (!this.data) return;

    this._drawBase(this.plain);
    this._drawBase(this.mixed);
    this._composite();

    const wipeX = this.state.compare && this.hasOverlay() ? this.state.wipe * W : W;

    ctx.drawImage(this.plain.canvas, 0, 0, W, W);
    ctx.save();
    ctx.beginPath();
    ctx.rect(0, 0, wipeX, W);
    ctx.clip();
    ctx.drawImage(this.mixed.canvas, 0, 0, W, W);
    ctx.restore();

    if (this.state.grid) {
      const step = W / 16; // one line every 8 pixels = 80 m on the ground
      ctx.strokeStyle = "rgba(10,20,30,.24)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let i = 1; i < 16; i++) {
        ctx.moveTo(i * step, 0); ctx.lineTo(i * step, W);
        ctx.moveTo(0, i * step); ctx.lineTo(W, i * step);
      }
      ctx.stroke();
    }
  }

  /** Screen coordinates -> which cell of the 128x128 grid was hit. */
  probe(clientX, clientY) {
    if (!this.data) return null;
    const rect = this.canvas.getBoundingClientRect();
    const x = Math.floor(((clientX - rect.left) / rect.width) * SIZE);
    const y = Math.floor(((clientY - rect.top) / rect.height) * SIZE);
    if (x < 0 || y < 0 || x >= SIZE || y >= SIZE) return null;
    const i = y * SIZE + x;
    const d = this.data;
    return {
      x, y,
      pred: d.pred ? d.pred[i] : null,
      truth: d.truth ? d.truth[i] : null,
      conf: d.conf ? d.conf[i] / 255 : null,
      metres: [x * 10, y * 10],
    };
  }

  /** The prediction alone, in crop colours -- used as the map overlay. */
  predictionDataUrl(opacity = 1) {
    if (!this.data || !this.data.pred) return null;
    const c = document.createElement("canvas");
    c.width = SIZE;
    c.height = SIZE;
    const ctx = c.getContext("2d");
    const img = ctx.createImageData(SIZE, SIZE);
    for (let i = 0; i < SIZE * SIZE; i++) {
      const cls = this.data.pred[i];
      const [r, g, b] = this.palette[cls] || [0, 0, 0];
      img.data[i * 4] = r;
      img.data[i * 4 + 1] = g;
      img.data[i * 4 + 2] = b;
      img.data[i * 4 + 3] = Math.round(255 * opacity);
    }
    ctx.putImageData(img, 0, 0);
    return c.toDataURL("image/png");
  }
}
