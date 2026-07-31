"""Director Control Room server.

Serves dashboard.html and exposes a small JSON API so the HTML page can:
  * read / write config.json
  * write the custom story file
  * start / stop a Director pipeline run (subprocess)
  * watch live console output + QC report + ComfyUI/Ollama health

Uses only the Python standard library (http.server + subprocess + threading).
Requires requests + numpy + imageio(-ffmpeg) installed in the .venv for the
actual Director run (same as running director.py yourself).

Usage:
    python dashboard_server.py [--port 8765] [--open]

Then open http://127.0.0.1:8765 in a browser.
"""
import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
DASHBOARD_PATH = os.path.join(BASE_DIR, "dashboard.html")
PYTHON = sys.executable

# ---------------------------------------------------------------- run state
_run = {
    "proc": None,
    "thread": None,
    "stdout": deque(maxlen=2000),   # list of lines
    "started": None,
    "stop_requested": False,
}
_run_lock = threading.Lock()


def _read_config() -> dict:
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _write_config(cfg: dict) -> None:
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def _out_dir() -> str:
    return _read_config().get("director", {}).get("output_dir", "output")


def _load_video_score():
    """Import director/video_score.py robustly regardless of cwd."""
    import importlib.util
    path = os.path.join(BASE_DIR, "video_score.py")
    spec = importlib.util.spec_from_file_location("director_video_score", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _health() -> dict:
    comfyui = ollama = False
    try:
        import requests
        try:
            comfyui = requests.get(_read_config().get("comfyui", {})
                                   .get("url", "http://127.0.0.1:8188") + "/system_stats",
                                   timeout=3).ok
        except Exception:
            comfyui = False
        try:
            url = _read_config().get("llm", {}).get("ollama_url",
                                                    "http://127.0.0.1:11434")
            ollama = requests.get(url + "/api/ps", timeout=3).ok
        except Exception:
            ollama = False
    except Exception:
        pass
    with _run_lock:
        running = _run["proc"] is not None and _run["proc"].poll() is None
    if not running:
        running = _director_running()
    return {"comfyui": comfyui, "ollama": ollama, "running": running}


_scan_cache = {"t": 0.0, "val": False}


def _director_running() -> bool:
    """True if a director.py process is active anywhere (terminal or dashboard).
    Cached ~5s to avoid hammering the OS on every health poll."""
    now = time.time()
    if now - _scan_cache["t"] < 5:
        return _scan_cache["val"]
    val = False
    try:
        if sys.platform.startswith("win"):
            cmd = ['powershell', '-NoProfile', '-Command',
                   "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                   "Where-Object { $_.CommandLine -match 'director\\.py' } | "
                   "Measure-Object | Select-Object -ExpandProperty Count"]
            out = subprocess.run(
                cmd, capture_output=True, text=True, timeout=10,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            val = out.stdout.strip() not in ("", "0")
        else:
            out = subprocess.run(["pgrep", "-f", r"director\.py"],
                                 capture_output=True, text=True, timeout=10)
            val = out.returncode == 0
    except Exception:
        val = False
    _scan_cache.update(t=now, val=val)
    return val


def _stdout_tail() -> str:
    """Read the shared director.log tail — covers runs launched from the
    terminal as well as from the dashboard."""
    cfg = _read_config()
    out_dir = cfg.get("director", {}).get("output_dir", "output")
    log_path = os.path.join(BASE_DIR, out_dir, "director.log")
    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 250_000))
            return f.read()
    except Exception:
        with _run_lock:
            return "\n".join(_run["stdout"])


def _report() -> dict:
    cfg = _read_config()
    out_dir = cfg.get("director", {}).get("output_dir", "output")
    rp = os.path.join(BASE_DIR, out_dir, "report.json")
    try:
        with open(rp, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _pump(proc: subprocess.Popen) -> None:
    """Drain subprocess stdout/stderr into the shared deque."""
    streams = []
    if proc.stdout:
        streams.append(proc.stdout)
    if proc.stderr:
        streams.append(proc.stderr)
    import select as _sel
    for s in streams:
        for line in iter(s.readline, b""):
            try:
                text = line.decode("utf-8", "replace").rstrip("\n")
            except Exception:
                text = ""
            if not text:
                continue
            with _run_lock:
                _run["stdout"].append(text)
            s.flush()
    proc.wait()


def _runner(script_args: list, stop_flag: threading.Event) -> None:
    cmd = [PYTHON, os.path.join(BASE_DIR, "director.py")] + script_args
    with _run_lock:
        _run["stdout"].clear()
        _run["stop_requested"] = False
        _run["started"] = time.time()
    try:
        proc = subprocess.Popen(
            cmd, cwd=BASE_DIR, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, bufsize=1,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except Exception as exc:
        with _run_lock:
            _run["stdout"].append(f"[controller] failed to start: {exc}")
        return
    with _run_lock:
        _run["proc"] = proc
    # waiter thread: waits for stop signal OR process end
    def _watch():
        # poll stop_flag; if set, terminate
        while proc.poll() is None:
            if stop_flag.is_set():
                try:
                    proc.terminate()
                except Exception:
                    pass
                break
            time.sleep(0.2)
    threading.Thread(target=_watch, daemon=True).start()
    _pump(proc)
    with _run_lock:
        _run["proc"] = None


def start_run(scene=None) -> str:
    with _run_lock:
        if _run["proc"] is not None and _run["proc"].poll() is None:
            return "A pipeline run is already active."
    stop_flag = threading.Event()
    args = ["--config", "config.json"]
    if scene is not None:
        args += ["--scene", str(scene)]
    t = threading.Thread(target=_runner, args=(args, stop_flag), daemon=True)
    with _run_lock:
        _run["thread"] = t
        _run["_stop_flag"] = stop_flag
    t.start()
    return ""


def stop_run() -> None:
    with _run_lock:
        flag = _run.get("_stop_flag")
    if flag is not None:
        flag.set()


def run_char_setup(no_refs: bool = False, timeout: int = 1800) -> dict:
    """Run characters.py --setup (blocking) and return its captured output.

    Runs in the request thread (the server is threaded), so the dashboard
    can trigger lock/reference generation and show the full log.
    """
    cmd = [PYTHON, os.path.join(BASE_DIR, "characters.py"),
           "--config", "config.json", "--setup"]
    if no_refs:
        cmd.append("--no-refs")
    try:
        out = subprocess.run(
            cmd, cwd=BASE_DIR, capture_output=True, text=True,
            timeout=timeout,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        text = ((out.stdout or "") + (out.stderr or "")).strip()
        return {"ok": out.returncode == 0, "exit_code": out.returncode,
                "output": text or "(no output)"}
    except subprocess.TimeoutExpired as exc:
        partial = exc.stdout or b""
        if isinstance(partial, bytes):
            partial = partial.decode("utf-8", "replace")
        return {"ok": False, "exit_code": -1,
                "output": str(partial).strip() +
                f"\n[char] timed out after {timeout}s"}
    except Exception as exc:
        return {"ok": False, "exit_code": -1, "output": str(exc)}


def clear_log() -> None:
    """Empty the shared director.log and the in-memory console queue."""
    cfg = _read_config()
    out_dir = cfg.get("director", {}).get("output_dir", "output")
    log_path = os.path.join(BASE_DIR, out_dir, "director.log")
    try:
        with open(log_path, "w", encoding="utf-8"):
            pass
    except Exception:
        pass
    with _run_lock:
        _run["stdout"].clear()


# ---------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path, content_type):
        try:
            with open(path, "rb") as f:
                body = f.read()
        except Exception:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):  # quiet
        pass

    # ---- GET ----
    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._file(DASHBOARD_PATH, "text/html; charset=utf-8")
        elif path == "/api/config":
            self._json({"config": _read_config()})
        elif path == "/api/health":
            h = _health()
            self._json({"status": {
                **h,
                "stdout": _stdout_tail(),
                "report": _report(),
            }})
        elif path == "/api/status":
            self._json({"status": {
                **_health(),
                "stdout": _stdout_tail(),
                "report": _report(),
            }})
        elif path == "/api/video_scores":
            vs_path = os.path.join(BASE_DIR, _out_dir(), "video_scores.json")
            if os.path.exists(vs_path):
                try:
                    with open(vs_path, encoding="utf-8") as f:
                        self._json(json.load(f))
                    return
                except Exception:
                    pass
            self._json({})
        elif path.startswith("/output/"):
            rel = path[len("/output/"):].replace("\\", "/")
            cfg = _read_config()
            out_dir = cfg.get("director", {}).get("output_dir", "output")
            base = os.path.abspath(os.path.join(BASE_DIR, out_dir))
            target = os.path.abspath(os.path.join(base, rel))
            if not target.startswith(base):
                self.send_error(403)
                return
            if target.endswith(".mp4"):
                self._file(target, "video/mp4")
            elif target.endswith(".json"):
                self._file(target, "application/json")
            elif target.endswith(".png") or target.endswith(".jpg"):
                self._file(target, "image/png")
            else:
                self._file(target, "application/octet-stream")
        else:
            self._json({"error": "not found"}, 404)

    # ---- POST ----
    def do_POST(self):
        path = self.path.split("?", 1)[0]
        try:
            n = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(n) if n else b"{}"
            data = json.loads(raw.decode("utf-8")) if raw else {}
        except Exception:
            data = {}
        if path == "/api/config":
            cfg = data.get("config")
            if not isinstance(cfg, dict):
                self._json({"error": "config object required"}, 400)
                return
            try:
                _write_config(cfg)
                self._json({"ok": True})
            except Exception as exc:
                self._json({"error": str(exc)}, 500)
        elif path == "/api/custom_story":
            text = str(data.get("text", ""))
            fname = str(data.get("file", "custom_story.txt")).replace("..", "")
            try:
                with open(os.path.join(BASE_DIR, fname), "w",
                          encoding="utf-8") as f:
                    f.write(text)
                self._json({"ok": True})
            except Exception as exc:
                self._json({"error": str(exc)}, 500)
        elif path == "/api/run":
            # optionally persist the incoming config first
            cfg = data.get("config")
            if isinstance(cfg, dict):
                try:
                    _write_config(cfg)
                except Exception as exc:
                    self._json({"error": str(exc)}, 500)
                    return
            scene = data.get("scene")
            scene = int(scene) if scene is not None else None
            msg = start_run(scene=scene)
            self._json({"ok": not msg, "message": msg or "started"},
                       200 if not msg else 409)
        elif path == "/api/stop":
            stop_run()
            self._json({"ok": True})
        elif path == "/api/char_setup":
            cfg = data.get("config")
            if isinstance(cfg, dict):
                try:
                    _write_config(cfg)
                except Exception as exc:
                    self._json({"error": str(exc)}, 500)
                    return
            no_refs = bool(data.get("no_refs"))
            result = run_char_setup(no_refs=no_refs)
            self._json(result, 200 if result["ok"] else 500)
        elif path == "/api/clear_log":
            clear_log()
            self._json({"ok": True})
        elif path == "/api/video_score":
            name = str(data.get("file", "")).replace("\\", "/")
            name = os.path.basename(name)
            if not name or ".." in name:
                self._json({"error": "invalid file"}, 400)
                return
            target = os.path.join(BASE_DIR, _out_dir(), name)
            try:
                vs = _load_video_score()
                result = vs.score_video(target)
            except Exception as exc:
                self._json({"error": str(exc)}, 500)
                return
            # cache this file's score into video_scores.json
            vs_path = os.path.join(BASE_DIR, _out_dir(), "video_scores.json")
            cache = {}
            try:
                with open(vs_path, encoding="utf-8") as f:
                    cache = json.load(f)
            except Exception:
                cache = {}
            cache[name] = result
            try:
                with open(vs_path, "w", encoding="utf-8") as f:
                    json.dump(cache, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
            self._json(result)
        elif path == "/api/video_score_all":
            try:
                vs = _load_video_score()
                results = vs.score_dir(os.path.join(BASE_DIR, _out_dir()))
            except Exception as exc:
                self._json({"error": str(exc)}, 500)
                return
            self._json(results)
        else:
            self._json({"error": "not found"}, 404)


def main():
    ap = argparse.ArgumentParser(description="Director Control Room server")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--open", action="store_true",
                    help="open the dashboard in a browser automatically")
    args = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"🎬 Director Control Room -> {url}")
    print("   Ctrl+C to stop the server.")
    if args.open:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        stop_run()
        print("\nBye.")


if __name__ == "__main__":
    main()
