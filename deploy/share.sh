set -euo pipefail

PORT="${PORT:-8000}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT}/deploy/logs"

if [ "${CROPSCOPE_AWAKE:-}" != "1" ] && command -v systemd-inhibit >/dev/null 2>&1; then
    export CROPSCOPE_AWAKE=1
    exec systemd-inhibit \
        --what=handle-lid-switch:sleep:idle \
        --who="CropScope" --why="public demo tunnel" \
        "${BASH_SOURCE[0]}" "$@"
fi

if ! command -v cloudflared >/dev/null 2>&1; then
    echo "cloudflared is missing. Install it with:  sudo pacman -S cloudflared" >&2
    exit 1
fi

mkdir -p "$LOG_DIR"
SERVER_LOG="${LOG_DIR}/server.log"
TUNNEL_LOG="${LOG_DIR}/tunnel.log"
: >"$SERVER_LOG"
: >"$TUNNEL_LOG"

SERVER_PID=""
TUNNEL_PID=""

cleanup() {
    echo
    echo "Taking the link offline..."
    [ -n "$TUNNEL_PID" ] && kill "$TUNNEL_PID" 2>/dev/null
    [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null
    wait 2>/dev/null
    return 0
}
trap cleanup EXIT INT TERM

cd "$ROOT"

echo "[1/2] Loading the model..."
uv run python -u webapp/server.py \
    --res_dir results --fold 1 --samples samples \
    --host 127.0.0.1 --port "$PORT" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

for _ in $(seq 90); do
    curl -sf "http://127.0.0.1:${PORT}/api/status" >/dev/null 2>&1 && break
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "The server stopped before it was ready:" >&2
        cat "$SERVER_LOG" >&2
        exit 1
    fi
    sleep 1
done
grep -E '^(Loaded|Model not loaded|Samples)' "$SERVER_LOG" | sed 's/^/      /' || true

echo "[2/2] Opening the tunnel..."
cloudflared tunnel --url "http://127.0.0.1:${PORT}" >"$TUNNEL_LOG" 2>&1 &
TUNNEL_PID=$!

URL=""
for _ in $(seq 60); do
    URL="$(grep -om1 'https://[a-z0-9-]*\.trycloudflare\.com' "$TUNNEL_LOG" || true)"
    [ -n "$URL" ] && break
    if ! kill -0 "$TUNNEL_PID" 2>/dev/null; then
        echo "The tunnel stopped before it gave out a URL:" >&2
        tail -20 "$TUNNEL_LOG" >&2
        exit 1
    fi
    sleep 1
done

if [ -z "$URL" ]; then
    echo "No URL after 60s. Look in ${TUNNEL_LOG}" >&2
    exit 1
fi

echo
echo "  ============================================================"
echo "     LIVE:  ${URL}"
echo "  ============================================================"
echo
echo "  Send that link to anyone. It works until you press Ctrl-C here."
echo "  This machine stays awake on its own -- closing the lid is safe."
echo
echo "  Logs:  ${SERVER_LOG}"
echo "         ${TUNNEL_LOG}"
echo

wait "$SERVER_PID" "$TUNNEL_PID"
