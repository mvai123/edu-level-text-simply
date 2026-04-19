#!/usr/bin/env sh
set -eu

ollama serve >/tmp/ollama.log 2>&1 &

python - <<'PY'
import time
import os
import requests

base = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
url = f"{base}/api/tags"
deadline = time.time() + 30
while time.time() < deadline:
    try:
        r = requests.get(url, timeout=2)
        if r.status_code == 200:
            break
    except Exception:
        pass
    time.sleep(0.5)
else:
    raise SystemExit(f"Ollama did not become ready at {base}")
PY

exec "$@"

