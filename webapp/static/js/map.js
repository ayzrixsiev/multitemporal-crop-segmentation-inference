/* The "where on Earth" panel.
 *
 * PASTIS ships the real geographic footprint of every patch, so a prediction can
 * be laid back down on the ground it came from. Leaflet is pulled from a CDN on
 * first use; if there is no internet the panel simply stays hidden rather than
 * breaking the rest of the page.
 */

const LEAFLET_CSS = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.css";
const LEAFLET_JS = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.js";

let loading = null;

function loadLeaflet() {
  if (window.L) return Promise.resolve(window.L);
  if (loading) return loading;

  loading = new Promise((resolve, reject) => {
    const css = document.createElement("link");
    css.rel = "stylesheet";
    css.href = LEAFLET_CSS;
    document.head.appendChild(css);

    const js = document.createElement("script");
    js.src = LEAFLET_JS;
    js.onload = () => resolve(window.L);
    js.onerror = () => reject(new Error("offline"));
    document.head.appendChild(js);

    setTimeout(() => reject(new Error("timeout")), 8000);
  }).catch((e) => {
    loading = null;
    throw e;
  });
  return loading;
}

export class PatchMap {
  constructor(hostId) {
    this.hostId = hostId;
    this.map = null;
    this.layer = null;
    this.frame = null;
  }

  async show({ bounds, polygon, overlayUrl, opacity = 0.75 }) {
    if (!bounds && !polygon) return false;
    let L;
    try {
      L = await loadLeaflet();
    } catch (e) {
      return false;
    }

    // bounds arrive as [south, west, north, east]
    const ring = polygon && polygon.length ? polygon : null;
    const bbox = bounds
      ? [[bounds[0], bounds[1]], [bounds[2], bounds[3]]]
      : L.latLngBounds(ring.map((p) => [p[0], p[1]])).pad(0);

    if (!this.map) {
      this.map = L.map(this.hostId, {
        zoomControl: true,
        attributionControl: true,
        scrollWheelZoom: false,
      });
      L.tileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        { maxZoom: 18, attribution: "Esri, Maxar, Earthstar Geographics" }
      ).addTo(this.map);
    }

    if (this.layer) this.map.removeLayer(this.layer);
    if (this.frame) this.map.removeLayer(this.frame);

    if (overlayUrl) {
      this.layer = L.imageOverlay(overlayUrl, bbox, { opacity, interactive: false });
      this.layer.addTo(this.map);
    }
    this.frame = L.rectangle(bbox, {
      color: "#0a9d7c",
      weight: 1.5,
      fill: false,
      dashArray: "4 3",
    }).addTo(this.map);

    this.map.fitBounds(bbox, { padding: [34, 34], maxZoom: 14 });
    setTimeout(() => this.map.invalidateSize(), 60);
    return true;
  }

  setOpacity(v) {
    if (this.layer) this.layer.setOpacity(v);
  }
}
