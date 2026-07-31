"""
director.py — The Director. End-to-end story -> image -> video -> QC pipeline.

Pipeline for a full production:
  1. Environment check   (ComfyUI alive? template + knobs present? models on disk?)
  2. Story               (custom_story.txt bypass  ->  LLM  ->  template fallback)
  3. Per scene:
       apply knobs (prompts + fresh seeds)
       -> submit scene workflow to ComfyUI
       -> wait for render
       -> download the 5s clip
       -> QUALITY CONTROL (duration / black / frozen / motion)
       -> PASS   -> keep clip, move to next scene
       -> FAIL   -> re-shoot with new seeds (up to qc.max_attempts)
       -> (optional) manual human review pause
  4. Stitch all accepted clips into one film (ffmpeg concat)
  5. Write a final production report (output/report.json)

Usage:
    python director.py              # full production
    python director.py --check      # environment check only (no rendering)
    python director.py --scene 2    # render scene index 2 only
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from typing import Optional

from comfy_api import ComfyUI, ComfyUIError
from qc import QualityChecker
from storywriter import StoryWriter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class _Tee:
    """Duplicates printed output to both the console and a log file so the
    dashboard server can tail a run launched from anywhere."""

    def __init__(self, stream, log_path):
        self._stream = stream
        self._log = open(log_path, "a", encoding="utf-8", buffering=1)

    def write(self, text):
        self._stream.write(text)
        self._stream.flush()
        self._log.write(text)
        return len(text)

    def flush(self):
        self._stream.flush()
        self._log.flush()


def _install_log_tee(config: dict) -> str:
    out_dir = config.get("director", {}).get("output_dir", "output")
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "director.log")
    sys.stdout = _Tee(sys.stdout, log_path)
    sys.stderr = _Tee(sys.stderr, log_path)
    return log_path


# Curated cinematic camera moves. `render.camera_move` picks one; a phrase is
# prepended to every video prompt to guarantee visible motion (our #1 quality
# factor). "auto"/"none" leave the prompt untouched.
CAMERA_MOVES = {
    "auto": "",
    "none": "",
    "dolly_in": "The camera dollies steadily in toward the subject, ",
    "push_in": "The camera pushes in close to the subject, ",
    "orbit": "The camera slowly orbits the subject clockwise, ",
    "orbit_ccw": "The camera slowly orbits the subject counter-clockwise, ",
    "pan_left": "The camera pans left across the scene, ",
    "pan_right": "The camera pans right across the scene, ",
    "tilt_up": "The camera tilts up from the subject, ",
    "tilt_down": "The camera tilts down toward the subject, ",
    "handheld": "A handheld camera drifts and sways subtly, ",
    "tracking": "The camera tracks alongside the moving subject, ",
    "crane_up": "The camera cranes upward to a wide reveal, ",
    "zoom_in": "The lens zooms in slowly, ",
}

_DEFAULT_BASE_SIGMAS = ("1.0, 0.99375, 0.9875, 0.98125, 0.975, "
                        "0.909375, 0.725, 0.421875, 0.0")
_DEFAULT_REFINE_SIGMAS = "0.85, 0.7250, 0.4219, 0.0"


def load_json(rel: str) -> dict:
    path = os.path.join(BASE_DIR, rel)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def apply_knobs(template: dict, knobs: dict, values: dict) -> dict:
    """Deep-copy the template and overwrite the knob inputs with scene values.

    A knob's value is either [node_id, input_name] or a list of such pairs
    (multi-path) so one setting (e.g. video_width) can patch several nodes.
    """
    wf = json.loads(json.dumps(template))
    for knob, path in knobs.items():
        if knob not in values:
            continue
        paths = path if path and isinstance(path[0], list) else [path]
        for node_id, inp in paths:
            if node_id in wf and inp in wf[node_id]["inputs"]:
                wf[node_id]["inputs"][inp] = values[knob]
    return wf


class Director:
    def __init__(self, config: dict) -> None:
        self.cfg = config
        self.comfy = ComfyUI(
            url=config["comfyui"]["url"],
            timeout=float(config["comfyui"].get("timeout_seconds", 30)),
        )
        self.qc = QualityChecker(config)
        self.writer = StoryWriter(config, BASE_DIR)
        dir_cfg = config["director"]
        self.output_dir = os.path.join(BASE_DIR, dir_cfg.get("output_dir", "output"))
        os.makedirs(self.output_dir, exist_ok=True)
        self.template = load_json(dir_cfg.get("workflow_template",
                                              "workflow_scene_template.json"))
        knobs_data = load_json(dir_cfg.get("workflow_knobs", "workflow_knobs.json"))
        self.knobs = knobs_data["knobs"]
        self.meta = knobs_data.get("meta", {})
        self.render_cfg = config.get("render", {})
        self.qc_cfg = config.get("qc", {})
        # Optional text appended to every VIDEO prompt to keep clips moving so
        # they pass QC motion checks even when the LLM writes slow cinematic
        # scenes. Set render.video_motion_boost to "" to disable.
        self.motion_boost = self.render_cfg.get("video_motion_boost", "").strip()

    # ------------------------------------------------------------- env check
    def check_environment(self) -> list:
        problems = []
        print("== DIRECTOR environment check ==")
        if not self.comfy.is_alive():
            problems.append("ComfyUI is not reachable at "
                            f"{self.cfg['comfyui']['url']}. Start ComfyUI Desktop "
                            "first (Settings -> enable API).")
        else:
            print(f"  ComfyUI OK: {self.cfg['comfyui']['url']}")
            missing_models = self._check_models()
            problems += missing_models
        for key in ("workflow_template", "workflow_knobs"):
            p = os.path.join(BASE_DIR, self.cfg["director"][key])
            if not os.path.exists(p):
                problems.append(f"missing {key}: {p}")
            else:
                print(f"  {key} OK")
        return problems

    def _check_models(self) -> list:
        """Verify that every model/widget referenced by the template exists on
        the ComfyUI server (via /object_info class listings)."""
        problems = []
        try:
            info = self.comfy.get("/object_info")
        except Exception as e:  # noqa: BLE001
            return [f"could not query /object_info: {e}"]
        # class -> input name whose value is a model filename
        loader_inputs = {
            "CheckpointLoaderSimple": "ckpt_name",
            "UNETLoader": "unet_name",
            "CLIPLoader": "clip_name",
            "VAELoader": "vae_name",
            "LTXAVTextEncoderLoader": "text_encoder",
            "LTXVAudioVAELoader": "ckpt_name",
            "LoraLoaderModelOnly": "lora_name",
            "LatentUpscaleModelLoader": "model_name",
        }
        for node in self.template.values():
            cls = node.get("class_type")
            inp_name = loader_inputs.get(cls)
            if not inp_name:
                continue
            val = node.get("inputs", {}).get(inp_name)
            if not isinstance(val, str):
                continue
            options = info.get(cls, {}).get("input", {}).get("required", {})
            # ComfyUI shape: required[name] == [ [choices...], {options} ]
            spec = options.get(inp_name) or []
            choices = spec[0] if spec and isinstance(spec[0], list) else []
            if choices and val not in choices:
                problems.append(f"{cls} '{val}' not found on server")
        return problems

    # ------------------------------------------------------------- rendering
    def _scene_values(self, scene: dict, scene_idx: int) -> dict:
        image_prompt = scene["image_prompt"]
        video_prompt = scene["video_prompt"]
        # Optional global character descriptor: applied to the keyframe image
        # (and video) so the same subject persists across every scene.
        char = str(self.cfg.get("story", {}).get("character", "") or "").strip()
        if char and not image_prompt.lower().startswith(char.lower()):
            image_prompt = f"{char}, {image_prompt}"
            if not video_prompt.lower().startswith(char.lower()):
                video_prompt = f"{char}, {video_prompt}"
        # Optional cinematic camera move: guarantees visible motion.
        camera = str(self.render_cfg.get("camera_move", "auto") or "auto")
        cam_phrase = CAMERA_MOVES.get(camera.strip().lower(), "")
        if cam_phrase:
            video_prompt = cam_phrase + video_prompt.lstrip()
        if self.motion_boost:
            video_prompt = f"{video_prompt.rstrip()}, {self.motion_boost}"
        return {
            "image_prompt": image_prompt,
            "image_seed": random.randint(0, 2**32 - 1),
            "video_prompt": video_prompt,
            "video_seed": random.randint(0, 2**32 - 1),
            "save_prefix": f"director/scene_{scene_idx:02d}",
            # The dashboard seconds_per_scene is authoritative; per-scene
            # duration_seconds from the story/LLM/fallback must NOT override it.
            "video_duration": float(
                self.cfg["story"].get("seconds_per_scene", 5)),
            "video_width": int(self.render_cfg.get("video_width", 480)),
            "video_height": int(self.render_cfg.get("video_height", 280)),
            "video_fps": int(self.render_cfg.get("fps", 25)),
            # LTX two-pass quality settings (empty/missing sigmas -> defaults).
            "video_base_strength": float(self.render_cfg.get(
                "video_base_strength", 0.7)),
            "video_base_sigmas": (self.render_cfg.get("video_base_sigmas")
                                  or "").strip() or _DEFAULT_BASE_SIGMAS,
            "video_refine_strength": float(self.render_cfg.get(
                "video_refine_strength", 1.0)),
            "video_refine_sigmas": (self.render_cfg.get("video_refine_sigmas")
                                    or "").strip() or _DEFAULT_REFINE_SIGMAS,
        }

    def render_scene(self, scene: dict, scene_idx: int) -> tuple:
        """Render one scene with QC retry loop.
        Returns (video_path, attempts, qc_result)."""
        max_attempts = int(self.qc_cfg.get("max_attempts", 3))
        qc_enabled = bool(self.qc_cfg.get("enabled", True))
        expected = float(self.cfg["story"]["seconds_per_scene"])
        out_name = f"scene_{scene_idx:02d}.mp4"

        for attempt in range(1, max_attempts + 1):
            values = self._scene_values(scene, scene_idx)
            wf = apply_knobs(self.template, self.knobs, values)
            print(f"\n--- scene {scene_idx} | attempt {attempt}/{max_attempts} ---")
            print(f"  image: {scene['image_prompt'][:80]}...")
            print(f"  video: {scene['video_prompt'][:80]}...")

            prompt_id = self.comfy.submit(wf)
            print(f"  submitted prompt_id={prompt_id} — rendering…")
            start = time.time()
            timeout = float(self.cfg["comfyui"].get(
                "render_timeout_seconds", 3600))
            if timeout <= 0:
                timeout = 3600  # 0/negative would time out instantly
            self.comfy.wait(prompt_id, timeout=timeout)
            elapsed = time.time() - start
            print(f"  render done in {elapsed:.0f}s")

            src = self._find_output(prompt_id)
            if src is None:
                print("  !! no output video file found in ComfyUI history")
                continue
            dest = os.path.join(self.output_dir, out_name)
            self.comfy.download(src["filename"], dest, src.get("subfolder", ""),
                                src.get("type", "output"))
            print(f"  downloaded -> {os.path.relpath(dest, BASE_DIR)}")

            if not qc_enabled:
                return dest, attempt, None
            result = self.qc.analyze(dest, expected)
            print(f"  QC: {'PASS' if result.passed else 'FAIL'}")
            print(f"       {result.summary()}")
            if result.passed:
                return dest, attempt, result
            print("  QC FAILED: " + "; ".join(result.reasons) +
                  " — re-shooting with new seeds")
        # exhausted attempts: keep last file, record failure
        last = os.path.join(self.output_dir, out_name)
        if os.path.exists(last):
            return last, max_attempts, None
        return "", max_attempts, None

    def _find_output(self, prompt_id: str) -> Optional[dict]:
        files = self.comfy.output_files(prompt_id)
        if not files:
            return None
        save_node = self.meta.get("output_video_node")
        for f in files:
            if save_node and str(f.get("node_id")) == str(save_node):
                return f
        return files[0]  # fall back to any output file

    # ------------------------------------------------------------- stitch
    def stitch(self, scene_paths: list, film_name: str = "final_film.mp4") -> str:
        # Paths may be relative to BASE_DIR (as stored in the report); resolve
        # them against BASE_DIR so stitching works no matter the CWD.
        resolved = []
        for p in scene_paths:
            if not p:
                continue
            full = p if os.path.isabs(p) else os.path.join(BASE_DIR, p)
            if os.path.exists(full):
                resolved.append(full)
        if not resolved:
            print("!! no accepted clips to stitch")
            return ""
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:  # noqa: BLE001
            ffmpeg = "ffmpeg"
        list_file = os.path.join(self.output_dir, "_concat.txt")
        with open(list_file, "w", encoding="utf-8") as f:
            for p in resolved:
                f.write(f"file '{p.replace(chr(39), chr(39)*4)}'\n")
        out = os.path.join(self.output_dir, film_name)
        fps = int(self.render_cfg.get("fps", 25))
        cmd = [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", list_file,
               "-r", str(fps), "-c:v", "libx264", "-crf", "18",
               "-pix_fmt", "yuv420p", "-an", out]
        print("\n== stitching film ==")
        print("  " + " ".join(cmd))
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=1200)
            return out
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"!! stitch failed: {e}")
            return ""

    # ------------------------------------------------------------- main run
    def run(self, only_scene: Optional[int] = None) -> None:
        problems = self.check_environment()
        if problems:
            print("\nENVIRONMENT PROBLEMS:")
            for p in problems:
                print("  - " + p)
            if only_scene is None:
                print("Full run aborted. Fix the problems above, then re-run "
                      "(or use --scene N to render a single scene anyway).")
                return

        story = self.writer.get_story()
        title = story["story_title"]
        scenes = story["scenes"]
        print(f"\n== DIRECTOR shooting: {title!r} ({len(scenes)} scenes) ==")

        if only_scene is not None:
            if not (0 <= only_scene < len(scenes)):
                print(f"!! scene index {only_scene} out of range "
                      f"(0..{len(scenes)-1})")
                return
            scenes = [scenes[only_scene]]
            # keep original index for filename
            base_idx = only_scene
        else:
            base_idx = 0

        results = []
        manual_review = bool(self.cfg["director"].get("manual_review", False))
        for i, scene in enumerate(scenes):
            idx = base_idx + i
            print(f"\n{'='*60}\nSCENE {idx+1}: {scene['video_prompt'][:60]}…\n"
                  f"{'='*60}")
            path, attempts, qc_result = self.render_scene(scene, idx)
            passed = qc_result is not None and qc_result.passed
            results.append({
                "scene": idx + 1,
                "attempts": attempts,
                "qc_passed": passed,
                "qc": qc_result.metrics if qc_result else {"unchecked": True},
                "file": os.path.relpath(path, BASE_DIR) if path else "",
                "image_prompt": scene["image_prompt"],
                "video_prompt": scene["video_prompt"],
                "audio_lines": scene.get("audio_lines", ""),
            })
            if manual_review and passed:
                input(f"  Human review: press Enter to accept scene {idx+1} "
                      "or Ctrl+C to abort…")

        report = {
            "story_title": title,
            "config": self.cfg,
            "scenes": results,
        }
        report_path = os.path.join(self.output_dir, "report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        if only_scene is None:
            film = self.stitch([r["file"] for r in results])
            report["final_film"] = os.path.relpath(film, BASE_DIR) if film else ""
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(f"\n== film saved -> {report.get('final_film', 'N/A')}")
        else:
            print(f"\n== scene rendered -> {results[0]['file']}")

        print(f"== report saved -> {os.path.relpath(report_path, BASE_DIR)}")
        n_pass = sum(1 for r in results if r["qc_passed"])
        print(f"== QC summary: {n_pass}/{len(results)} scenes passed QC")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="ComfyUI Director Mode")
    parser.add_argument("--config", default="config.json",
                        help="path to config.json (default: config.json)")
    parser.add_argument("--check", action="store_true",
                        help="run environment/model check only")
    parser.add_argument("--scene", type=int, default=None,
                        help="render only one scene index (0-based)")
    args = parser.parse_args(argv)

    config = load_json(args.config)
    log_path = _install_log_tee(config)
    print(f"== log -> {os.path.relpath(log_path, BASE_DIR)}")
    director = Director(config)
    if args.check:
        problems = director.check_environment()
        print("\n" + ("ALL CHECKS PASSED" if not problems
                      else "\n".join("- " + p for p in problems)))
        return
    director.run(only_scene=args.scene)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nDirector stopped by user.")
        sys.exit(130)
    except ComfyUIError as e:
        print(f"\nDIRECTOR ERROR: {e}")
        sys.exit(1)
