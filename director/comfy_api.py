"""
comfy_api.py — Minimal ComfyUI API client for the Director app.

Drives a headless ComfyUI server (Desktop or manual install) over its HTTP API:
  * GET  /system_stats          -> server alive check
  * POST /prompt                -> submit a flattened API-format workflow
  * GET  /history/{prompt_id}   -> poll for completion + output filenames
  * GET  /view?filename=...     -> download a rendered artifact

No websocket dependency: progress is polled via /history, which is simple,
robust, and plenty fast for a per-scene pipeline.
"""
from __future__ import annotations

import time
import uuid
from typing import Any, Optional

import requests

POLL_INTERVAL_S = 1.0
GB = 1024 ** 3


def _gb(n: float) -> str:
    """Format a byte count as a readable GiB string (tolerant of bad input)."""
    try:
        return f"{float(n) / GB:.1f} GB"
    except Exception:  # noqa: BLE001
        return "?"


class ComfyUIError(RuntimeError):
    pass


class ComfyUI:
    def __init__(self, url: str = "http://127.0.0.1:8188",
                 client_id: Optional[str] = None,
                 timeout: float = 30.0) -> None:
        self.url = url.rstrip("/")
        self.client_id = client_id or str(uuid.uuid4())
        self.timeout = timeout
        self.session = requests.Session()

    # ------------------------------------------------------------- server
    def is_alive(self) -> bool:
        try:
            r = self.session.get(f"{self.url}/system_stats", timeout=5)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def system_stats(self) -> dict:
        r = self.session.get(f"{self.url}/system_stats", timeout=10)
        r.raise_for_status()
        return r.json()

    def free_memory(self, unload_models: bool = True,
                    free_memory: bool = True) -> bool:
        """Ask ComfyUI to unload model weights (VRAM) and free cached RAM.

        POST /free is ComfyUI's own cleanup route: it drops model weights from
        VRAM and reclaims cached RAM so the next job starts from a fresh state.
        Call it between scenes/segments to stop memory from accumulating on
        low-VRAM cards. Never raises - cleanup is best-effort."""
        try:
            r = self.session.post(
                f"{self.url}/free",
                json={"unload_models": unload_models,
                      "free_memory": free_memory},
                timeout=15)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def memory_summary(self) -> Optional[str]:
        """Human-readable 'VRAM x / y free; RAM z / w free' summary from
        /system_stats, or None when the server does not report metrics."""
        try:
            stats = self.system_stats()
        except Exception:  # noqa: BLE001
            return None
        devs = stats.get("devices") or []
        gpu = devs[0] if devs else {}
        bits = []
        vtotal = gpu.get("vram_total") or 0
        vfree = gpu.get("vram_free") or 0
        if vtotal:
            bits.append(f"VRAM {_gb(vfree)}/{_gb(vtotal)} free")
        syst = stats.get("system") or {}
        rtotal = syst.get("ram_total") or 0
        rfree = syst.get("ram_free") or 0
        if rtotal:
            bits.append(f"RAM {_gb(rfree)}/{_gb(rtotal)} free")
        return "; ".join(bits) if bits else None

    # ------------------------------------------------------------- submit
    def free_ram_gb(self) -> Optional[float]:
        """Current free system RAM in GB (from /system_stats), or None when
        the server does not report it / is unreachable."""
        try:
            stats = self.system_stats()
        except Exception:  # noqa: BLE001
            return None
        syst = stats.get("system") or {}
        rfree = syst.get("ram_free") or 0
        return float(rfree) / GB if rfree else None

    # ------------------------------------------------------------- submit
    def submit(self, workflow: dict) -> str:
        """POST an API-format workflow; returns the prompt_id."""
        payload = {"prompt": workflow, "client_id": self.client_id}
        r = self.session.post(f"{self.url}/prompt", json=payload, timeout=30)
        if r.status_code != 200:
            raise ComfyUIError(f"POST /prompt failed ({r.status_code}): {r.text[:1000]}")
        data = r.json()
        if "prompt_id" not in data:
            raise ComfyUIError(f"Unexpected /prompt response: {data}")
        return data["prompt_id"]

    # ------------------------------------------------------------- polling
    def wait(self, prompt_id: str, timeout: float = 3600.0) -> dict:
        """Poll /history until the prompt finishes. Returns the history entry
        (keys: 'status', 'outputs'). Raises ComfyUIError on failure/timeout."""
        deadline = time.time() + timeout
        last = None
        while time.time() < deadline:
            try:
                r = self.session.get(f"{self.url}/history/{prompt_id}", timeout=10)
                if r.status_code == 200:
                    data = r.json()
                    if prompt_id in data:
                        last = data[prompt_id]
                        status = last.get("status", {})
                        if status.get("completed") or status.get("status_str") in (
                                "success", "error"):
                            if status.get("status_str") == "error" or status.get("messages"):
                                msgs = status.get("messages", [])
                                for _, m in msgs:
                                    if m.get("type") == "execution_error":
                                        raise ComfyUIError(
                                            f"execution_error: {m.get('message', {})}")
                            return last
            except requests.RequestException:
                pass  # server may be briefly busy; keep polling
            time.sleep(POLL_INTERVAL_S)
        raise ComfyUIError(f"Timeout waiting for prompt {prompt_id}")

    def wait_status(self, prompt_id: str, timeout: float = 3600.0) -> str:
        entry = self.wait(prompt_id, timeout)
        return entry.get("status", {}).get("status_str", "unknown")

    # ------------------------------------------------------------- outputs
    def history(self, prompt_id: str) -> dict:
        r = self.session.get(f"{self.url}/history/{prompt_id}", timeout=10)
        r.raise_for_status()
        data = r.json()
        return data.get(prompt_id, {})

    def output_files(self, prompt_id: str) -> list[dict]:
        """Collect all output file references from a finished prompt:
        [{'node_id': ..., 'filename': ..., 'subfolder': ..., 'type': ...}]"""
        entry = self.history(prompt_id)
        files: list[dict] = []
        for node_id, node_out in entry.get("outputs", {}).items():
            for img in node_out.get("images", []):
                files.append({"node_id": node_id, **img})
            for vids in node_out.get("videos", []):
                files.append({"node_id": node_id, **vids})
            for auds in node_out.get("audio", []):
                files.append({"node_id": node_id, **auds})
        return files

    # ------------------------------------------------------------- download
    def download(self, filename: str, dest_path: str,
                 subfolder: str = "", ftype: str = "output") -> str:
        params = {"filename": filename, "subfolder": subfolder, "type": ftype}
        r = self.session.get(f"{self.url}/view", params=params, timeout=60)
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            f.write(r.content)
        return dest_path

    def upload_image(self, image_path: str) -> dict:
        """Upload an image for use with LoadImage; returns the server reference."""
        with open(image_path, "rb") as f:
            files = {"image": (image_path.split("\\")[-1].split("/")[-1], f,
                               "image/png")}
            r = self.session.post(f"{self.url}/upload/image", files=files,
                                  timeout=60)
        r.raise_for_status()
        return r.json()

    def get(self, path: str, **params: Any) -> dict:
        r = self.session.get(f"{self.url}{path}", params=params, timeout=30)
        r.raise_for_status()
        return r.json()

    def post(self, path: str, payload: dict) -> dict:
        r = self.session.post(f"{self.url}{path}", json=payload, timeout=30)
        r.raise_for_status()
        return r.json()
