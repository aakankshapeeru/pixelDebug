import time
import urllib.request

for attempt in range(60):
    try:
        for url in ["http://127.0.0.1:8000/health", "http://127.0.0.1:5173"]:
            urllib.request.urlopen(url, timeout=1).close()
        break
    except OSError:
        time.sleep(0.5)
else:
    raise SystemExit("Local services did not become ready")
