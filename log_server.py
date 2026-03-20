"""
Minimal HTTP log viewer.
Serves trading.log at / on port 8080 so it can be read and copied from a browser.
"""
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

LOG_FILE = "trading.log"
PORT = 8080

_HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Trading Agent Logs</title>
  <meta http-equiv="refresh" content="30">
  <style>
    body {{ background:#0d0d0d; color:#c8ffc8; font-family:monospace; font-size:13px; margin:0; padding:16px; }}
    pre {{ white-space:pre-wrap; word-break:break-all; }}
    h2 {{ color:#7fff7f; margin-bottom:4px; }}
    small {{ color:#5a5; }}
  </style>
</head>
<body>
  <h2>Trading Agent — Live Log</h2>
  <small>Auto-refreshes every 30 seconds &nbsp;|&nbsp; <a href="/" style="color:#7fff7f">Refresh now</a></small>
  <hr style="border-color:#2a4a2a">
  <pre>{content}</pre>
</body>
</html>"""


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path not in ("/", "/logs"):
            self.send_response(404)
            self.end_headers()
            return

        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r", errors="replace") as f:
                content = f.read()
        else:
            content = "(No log file yet — waiting for first scan cycle…)"

        body = _HTML_TEMPLATE.format(content=content).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def start_log_server():
    server = HTTPServer(("0.0.0.0", PORT), _Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
