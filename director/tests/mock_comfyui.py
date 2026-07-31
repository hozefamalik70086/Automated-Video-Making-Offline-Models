"""
tests/mock_comfyui.py — A tiny fake ComfyUI server for testing the Director
pipeline WITHOUT a GPU or a real ComfyUI install.

It implements just enough of the ComfyUI HTTP API for the director to run:
    GET  /system_stats          -> server "alive"
    GET  /object_info           -> returns the loader classes + model filenames
                                   so the director's model check passes
    POST /prompt                -> accepts an API-format workflow, "renders" a
                                   synthetic mp4, records it in /history
    GET  /history/{id}          -> completed entry pointing at the video
    GET  /view                  -> serves the rendered video

Rendering behavior (deterministic retry test):
    * The 1st attempt of every scene produces a BLACK/STATIC video  -> QC FAIL
    * The 2nd+ attempt produces a MOVING BAR video                  -> QC PASS
    This forces the director to exercise its automatic re-shoot loop, then
    confirms the accepted clip passes QC and gets stitched into the film.

Run:
    python tests/mock_comfyui.py --port 8899
Then point a test config at http://127.0.0.1:8899 (see config_mock_*.json).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
import urllib.parse
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import imageio.v2 as imageio

BASE = os.path.dirname(os.path.abspath(__file__))
MOCK_OUT = os.path.join(BASE, "_mock_output")
os.makedirs(MOCK_OUT, exist_ok=True)

HISTORY: dict = {}
ATTEMPTS: dict = {}
IMG_ATTEMPTS: dict = {}
LOCK = threading.Lock()

# model filenames used by the flattened scene template (verified in --check)
_MODELS = {
    "z_image_turbo_int8_convrot.safetensors",
    "qwen_3_4b_fp8_mixed.safetensors",
    "ae.safetensors",
    "ltx-2.3-22b-dev-fp8.safetensors",
    "gemma_3_12B_it_fp4_mixed_2.safetensors",
    "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
    "ltx-2.3-22b-distilled-1.1_lora-dynamic_fro09_avg_rank_111_bf16.safetensors",
}


def _loader(cls: str, input_name: str, model: str) -> dict:
    return {cls: {"input": {"required": {input_name: [[model], {}]}}}}


def build_object_info() -> dict:
    info = {}
    info.update(_loader("CheckpointLoaderSimple", "ckpt_name",
                        "ltx-2.3-22b-dev-fp8.safetensors"))
    info.update(_loader("UNETLoader", "unet_name",
                        "z_image_turbo_int8_convrot.safetensors"))
    info.update(_loader("CLIPLoader", "clip_name", "qwen_3_4b_fp8_mixed.safetensors"))
    info.update(_loader("VAELoader", "vae_name", "ae.safetensors"))
    info.update(_loader("LTXAVTextEncoderLoader", "text_encoder",
                        "gemma_3_12B_it_fp4_mixed_2.safetensors"))
    info.update(_loader("LTXVAudioVAELoader", "ckpt_name",
                        "ltx-2.3-22b-dev-fp8.safetensors"))
    info.update(_loader("LoraLoaderModelOnly", "lora_name",
                        "ltx-2.3-22b-distilled-1.1_lora-dynamic_fro09_avg_rank_111_bf16.safetensors"))
    info.update(_loader("LatentUpscaleModelLoader", "model_name",
                        "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"))
    return info


INFO = build_object_info()


def write_video(path: str, good: bool) -> None:
    """Write a 5s @25fps synthetic clip. good=False -> black/static (QC fail)."""
    w, h, fps, n = 480, 288, 25, 125
    frames = []
    for i in range(n):
        f = np.zeros((h, w, 3), dtype=np.uint8)
        if good:
            x = (i * 6) % (w - 120)
            f[100:180, x:x + 120] = 200
        frames.append(f)
    imageio.mimsave(path, frames, fps=fps)


def write_image(path: str) -> None:
    """Write a deterministic synthetic PNG (for SaveImage / char-ref workflows)."""
    img = np.full((720, 1080, 3), 60, dtype=np.uint8)
    img[200:520, 300:780] = 180  # block in the middle so it isn't blank
    imageio.imwrite(path, img)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silence request logging
        pass

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ------------------------------------------------------------- GET
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        qs = urllib.parse.parse_qs(parsed.query)
        if path == "/system_stats":
            self._json(200, {"system": {"comfyui_version": "mock",
                                        "devices": [{"name": "synthetic"}]}})
        elif path == "/object_info":
            self._json(200, INFO)
        elif path.startswith("/history/"):
            pid = path.rsplit("/", 1)[-1]
            with LOCK:
                hist = dict(HISTORY)
            self._json(200, {pid: hist[pid]} if pid in hist else {})
        elif path == "/view":
            fname = qs.get("filename", [""])[0]
            sub = qs.get("subfolder", [""])[0]
            p = os.path.join(MOCK_OUT, sub, fname)
            if os.path.exists(p):
                with open(p, "rb") as f:
                    data = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_response(404)
                self.end_headers()
        else:
            self._json(404, {"error": f"no mock route {path}"})

    # ------------------------------------------------------------- POST
    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/prompt":
            self._json(404, {"error": f"no mock route {parsed.path}"})
            return
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        workflow = body.get("prompt", {})

        # Detect the save node: SaveVideo -> scene clip, SaveImage -> char ref
        prefix, node_id, output_kind = "scene", None, "video"
        for nid, node in workflow.items():
            cls = node.get("class_type")
            if cls == "SaveVideo":
                prefix = node.get("inputs", {}).get("filename_prefix", "scene")
                node_id = nid
                output_kind = "video"
                break
            if cls == "SaveImage":
                prefix = node.get("inputs", {}).get("filename_prefix", "img")
                node_id = nid
                output_kind = "image"
        m = re.search(r"scene_(\d+)", str(prefix))
        scene = int(m.group(1)) if m else 1

        if output_kind == "image":
            # character-reference workflows — always succeed, separate counter
            with LOCK:
                IMG_ATTEMPTS[prefix] = IMG_ATTEMPTS.get(prefix, 0) + 1
                attempt = IMG_ATTEMPTS[prefix]
            safe = re.sub(r"[^A-Za-z0-9_\-]+", "_", prefix)
            fname = f"{safe}_attempt{attempt}.png"
            write_image(os.path.join(MOCK_OUT, fname))
            outputs = {str(node_id): {"images": [
                {"filename": fname, "subfolder": "", "type": "output"}]}}
        else:
            with LOCK:
                ATTEMPTS[scene] = ATTEMPTS.get(scene, 0) + 1
                attempt = ATTEMPTS[scene]
            good = attempt >= 2  # 1st attempt always bad -> triggers retry
            fname = f"scene_{scene:02d}_attempt{attempt}.mp4"
            write_video(os.path.join(MOCK_OUT, fname), good)
            outputs = {str(node_id): {"videos": [
                {"filename": fname, "subfolder": "", "type": "output"}]}}

        pid = str(uuid.uuid4())
        with LOCK:
            HISTORY[pid] = {
                "status": {"status_str": "success", "completed": True,
                           "messages": []},
                "outputs": outputs,
            }
        kind = output_kind
        print(f"[mock] prompt {pid} scene {scene} attempt {attempt} "
              f"({kind}) -> "
              f"{'good' if kind == 'image' or good else 'BAD'}")
        self._json(200, {"prompt_id": pid, "number": len(HISTORY),
                         "node_errors": []})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"mock ComfyUI listening on http://{args.host}:{args.port}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("mock stopped")


if __name__ == "__main__":
    main()
