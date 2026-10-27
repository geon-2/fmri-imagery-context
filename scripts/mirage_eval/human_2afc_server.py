"""사람 평가 페이지를 띄우고 응답을 파일에 저장하는 로컬 서버(127.0.0.1 전용).
정답 키(key.json)는 site/ 밖에 있어 페이지에서 접근할 수 없다. 응답은 한 줄씩 append, 같은 문항은 마지막 응답이 유효.
사용: python scripts/mirage_eval/human_2afc_server.py [--port 8765] [--resp data/results/human_eval/responses.jsonl]"""
import argparse, functools, json, os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8765)
ap.add_argument("--site", default="data/results/human_eval/site")
ap.add_argument("--resp", default="data/results/human_eval/responses.jsonl")
a = ap.parse_args()

def latest():
    out = {}
    if os.path.exists(a.resp):
        for line in open(a.resp):
            if line.strip():
                r = json.loads(line); out[r["id"]] = r
    return out

class H(SimpleHTTPRequestHandler):
    def log_message(self, *args): pass
    def end_headers(self):
        self.send_header("Cache-Control", "no-store"); super().end_headers()
    def _json(self, obj, code=200):
        b = json.dumps(obj).encode(); self.send_response(code)
        self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b)))
        self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path.split("?")[0] == "/responses": return self._json(latest())
        super().do_GET()
    def do_POST(self):
        if self.path != "/save": return self._json({"error": "not found"}, 404)
        r = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if r.get("choice") not in ("L", "R") or not isinstance(r.get("id"), str): return self._json({"error": "bad"}, 400)
        rec = {"id": r["id"], "choice": r["choice"], "rt_ms": r.get("rt_ms"), "ts": r.get("ts")}
        os.makedirs(os.path.dirname(a.resp), exist_ok=True)
        with open(a.resp, "a") as f: f.write(json.dumps(rec) + "\n"); f.flush(); os.fsync(f.fileno())
        self._json({"ok": True, "n": len(latest())})

print(f"http://127.0.0.1:{a.port}  (응답 저장: {a.resp})", flush=True)
ThreadingHTTPServer(("127.0.0.1", a.port), functools.partial(H, directory=a.site)).serve_forever()
