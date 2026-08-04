"""
director.py — The Director. End-to-end story -> image -> video -> QC pipeline.

Pipeline for a full production:
  1. Environment check   (ComfyUI alive? template + knobs present? models on disk?)
  2. Story               (custom_story.txt bypass  ->  LLM  ->  template fallback)
  3. Per scene:
       apply knobs (prompts + fresh seeds)
       -> render the scene — either as ONE clip (render.chunked=false) or as
          N 1-second segments (render.chunked=true, default) which are joined
          right after creation. Chunking keeps every individual LTX generation
          tiny so high FPS + high resolution (e.g. 1080x1920 @ 30fps, 9:16)
          stay within VRAM and never hang on long/high-fps renders.
          With render.chain_frames, each segment continues the previous
          segment's last frame for a continuous shot.
       -> download the clip(s)
       -> QUALITY CONTROL (duration / black / frozen / motion) per segment
       -> PASS   -> keep clip, move to next scene/segment
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
import math
import os
import random
import shutil
import subprocess
import sys
import time
from typing import Optional

from comfy_api import ComfyUI, ComfyUIError
from characters import CharacterLibrary, CONSISTENCY_ANCHOR
from qc import QualityChecker, QCResult
from storywriter import StoryWriter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Known install folder (relative to <ComfyUI>/models/) for a model filename,
# used to turn a bare "not found" env-check message into an actionable hint.
_MODEL_INSTALL_FOLDERS = {
    "ae.safetensors": "vae",
    "z_image_turbo_int8_convrot.safetensors": "diffusion_models",
    "qwen_3_4b_fp8_mixed.safetensors": "text_encoders",
    "gemma_3_12B_it_fp4_mixed_2.safetensors": "text_encoders",
    "ltx-2.3-22b-dev-fp8.safetensors": "checkpoints",
    "ltx-2.3-22b-distilled-1.1_lora-dynamic_fro09_avg_rank_111_bf16.safetensors": "loras",
    "ltx-2.3-spatial-upscaler-x2-1.1.safetensors": "upscale_models",
}
# Optional direct download link for known models to save the user a search.
_MODEL_DOWNLOAD_URLS = {
    "ae.safetensors": "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors",
}


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

# Art-style keywords. If a scene's image prompt contains none of these, the
# director appends the config genre as a style phrase so the scene does NOT
# default to photorealistic (the LLM / custom story often only tags scene 1
# with the style, making scenes 2+ come out as real humans).
_ART_STYLE_HINTS = (
    "watercolor", "oil painting", "anime", "ghibli", "pixar",
    "3d render", "3d-render", "cgi", "photoreal", "photorealistic",
    "realistic photo", "realistic photography", "cartoon", "claymation",
    "pixel art", "concept art", "digital painting", "digital art",
    "illustration", "in the style", "style of", "aesthetic", "hand-drawn",
    "hand drawn", "acrylic", "gouache", "cel-shaded", "cel shaded",
    "stop motion", "low poly", "voxel", "comic book", "manga",
    "impressionist", "art nouveau", "art deco", "baroque", "ukiyo-e",
    "chibi", "toon",
)


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
        self.chars = CharacterLibrary(config, BASE_DIR)
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
                hint = ""
                folder = _MODEL_INSTALL_FOLDERS.get(val)
                if folder:
                    hint = f"  -> place it at <ComfyUI>/models/{folder}/{val}"
                url = _MODEL_DOWNLOAD_URLS.get(val)
                if url:
                    hint += f"  download: {url}"
                problems.append(f"{cls} '{val}' not found on server.{hint}")
        return problems

    # ------------------------------------------------------------- rendering
    def _genre_style_phrase(self) -> str:
        """The user's chosen film style from config story.genre, cleaned."""
        genre = str(self.cfg.get("story", {}).get("genre", "") or "").strip()
        return genre.rstrip(" .,;")

    def _ensure_style_phrase(self, prompt: str, sibling: str | None = None) -> str:
        """Append the configured genre/style phrase when neither prompt carries an art-style hint."""
        genre_style = self._genre_style_phrase()
        if not genre_style:
            return prompt
        lower = prompt.lower()
        if any(h in lower for h in _ART_STYLE_HINTS):
            return prompt
        if sibling and any(h in sibling.lower() for h in _ART_STYLE_HINTS):
            return prompt
        return f"{prompt.rstrip(' .,')}. {genre_style}"

    def _scene_values(self, scene: dict, scene_idx: int,
                      duration: Optional[float] = None,
                      chunk_idx: Optional[int] = None,
                      chunk_total: Optional[int] = None) -> dict:
        image_prompt = scene["image_prompt"]
        video_prompt = scene["video_prompt"]
        # Per-scene LOCKED character descriptors (Phase 3): append each
        # character's master visual descriptor + a consistency anchor to the
        # image prompt, and a short "same characters" note to the video prompt.
        image_blocks, video_note = [], ""
        chars_active = bool(getattr(self, "chars", None) and self.chars.enabled)
        if chars_active:
            image_blocks, video_note = self.chars.scene_character_blocks(scene)
        # Legacy global character descriptor: applied to the keyframe image
        # (and video) so one subject persists across every scene. SKIPPED when
        # character locking is active — the per-scene locked descriptors already
        # carry identity, and prepending the global subject here would inject an
        # unrelated character into every scene (e.g. a "toon girl" into a scene
        # that only contains an umbrella).
        if not chars_active:
            char = str(self.cfg.get("story", {}).get("character", "") or "").strip()
            if char and not image_prompt.lower().startswith(char.lower()):
                image_prompt = f"{char}, {image_prompt}"
                if not video_prompt.lower().startswith(char.lower()):
                    video_prompt = f"{char}, {video_prompt}"
        # Enforce a consistent art style on EVERY scene. The story/LLM often
        # only tags scene 1 with the style; untagged scenes otherwise default
        # to photorealistic (a real human). If neither prompt carries an
        # explicit art-style keyword, append the config genre as the style
        # phrase to both prompts so the motion clip preserves the same look.
        image_prompt = self._ensure_style_phrase(image_prompt, video_prompt)
        video_prompt = self._ensure_style_phrase(video_prompt, image_prompt)
        if image_blocks:
            image_prompt = (image_prompt.rstrip(" .,") + ". "
                            + " ".join(image_blocks) + " " + CONSISTENCY_ANCHOR)
        # Optional cinematic camera move: guarantees visible motion.
        camera = str(self.render_cfg.get("camera_move", "auto") or "auto")
        cam_phrase = CAMERA_MOVES.get(camera.strip().lower(), "")
        if cam_phrase:
            video_prompt = cam_phrase + video_prompt.lstrip()
        if self.motion_boost:
            video_prompt = f"{video_prompt.rstrip()}, {self.motion_boost}"
        if video_note and video_note not in video_prompt:
            video_prompt = f"{video_prompt.rstrip(' .,')}. {video_note}"
        return {
            "image_prompt": image_prompt,
            "image_seed": random.randint(0, 2**32 - 1),
            "video_prompt": video_prompt,
            "video_seed": random.randint(0, 2**32 - 1),
            # Chunked scenes save each 1s segment under its own prefix so the
            # per-segment clips survive alongside the joined scene file.
            "save_prefix": (f"director/scene_{scene_idx:02d}"
                            if chunk_idx is None
                            else f"director/scene_{scene_idx:02d}_c{chunk_idx:02d}"),
            # The dashboard seconds_per_scene (or story.video_length/scenes) is
            # authoritative; per-scene duration_seconds from the story/LLM/
            # fallback must NOT override it. With chunked rendering, `duration`
            # is the single small segment length.
            "video_duration": float(
                duration if duration is not None
                else self._effective_scene_seconds()),
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

    # ------------------------------------------------------------- rendering
    def _effective_scene_seconds(self) -> float:
        """Per-scene render duration (seconds).

        ``story.video_length`` (the total video length the user asks for) is
        authoritative when set: each scene gets ``video_length / scenes`` so a
        long runtime is automatically SPREAD across the scenes instead of every
        scene becoming the full length. Falls back to ``seconds_per_scene``.
        """
        story = self.cfg.get("story", {})
        total = float(story.get("video_length") or 0)
        if total and total > 0:
            n = max(1, int(story.get("scenes", 1)))
            return total / n
        return float(story.get("seconds_per_scene", 5))

    def _segment_seconds(self) -> float:
        """Effective single-generation length (seconds).

        The user's ``render.chunk_seconds`` is the *preferred* segment length,
        but it is always capped by the hard safety ceiling
        ``render.max_segment_seconds``. A single LTX generation therefore NEVER
        grows huge regardless of the config — this is what prevents OOM /
        "not enough memory" at 30-60s scenes on the RTX 4080."""
        chunk = max(0.1, float(self.render_cfg.get("chunk_seconds", 1.0)))
        cap = max(0.1, float(self.render_cfg.get("max_segment_seconds", 1.0)))
        return min(chunk, cap)

    def _chunk_plan(self, total_seconds: float) -> int:
        """Number of segments for a scene of this length."""
        return max(1, int(math.ceil(total_seconds / self._segment_seconds())))

    def _free_ram_gb(self) -> Optional[float]:
        """Free system RAM (GB) per ComfyUI /system_stats, or None if unknown
        / the client has no such method (guards offline unit-test fakes)."""
        if not hasattr(self.comfy, "free_ram_gb"):
            return None
        try:
            return self.comfy.free_ram_gb()
        except Exception:  # noqa: BLE001
            return None

    def _clean_memory(self, stage: str) -> None:
        """Release RAM/VRAM pressure between heavy generations.

        ``stage`` is 'segment' or 'scene'. Controlled by the ``comfyui``
        config section:
          * free_between_scenes  (default True)  - unload everything at scene
            boundaries so RAM/VRAM cannot accumulate across a long run (the
            LTX fp8 + 12B text encoder are both heavy).
          * free_between_segments (default False) - unload between every 1s
            segment. Disabled by default because reloading the I2V model for
            each tiny segment slows chunked runs.
          * min_free_ram_gb (default 6.0) - SAFETY NET: even when
            free_between_segments is OFF, if free system RAM drops below this
            we STILL unload ComfyUI's models. Without this, a long chunked
            scene (many 1s segments keeping LTX + 12B encoder resident)
            exhausts RAM and the whole machine freezes mid-scene (observed
            stalling at ~segment 10-12 of 12). The adaptive free only fires
            when RAM is genuinely tight, so healthy runs stay fast.
          * gc_collect (default True)              - forces Python GC so this
            process releases any large downloaded-file buffers. Runs
            regardless of the free toggles above.
        Never crashes the pipeline - every step is best-effort."""
        conf = self.cfg.get("comfyui", {})
        force = (bool(conf.get("free_between_scenes", True))
                 if stage == "scene"
                 else bool(conf.get("free_between_segments", False)))
        # Adaptive safety net: unload models even when segment-freeing is OFF
        # if the system is critically low on RAM, so we never freeze mid-scene.
        if not force and stage == "segment":
            floor = float(conf.get("min_free_ram_gb", 6.0))
            free = self._free_ram_gb()
            if free is not None and free < floor:
                print(f"  [mem] free RAM {free:.1f} GB < {floor:.1f} GB "
                      f"threshold; unloading models to avoid a freeze")
                force = True
        if force:
            try:
                if hasattr(self.comfy, "free_memory"):
                    if self.comfy.free_memory():
                        print(f"  [mem] freed ComfyUI models ({stage})")
            except Exception as exc:  # noqa: BLE001
                print(f"  [mem] !! free failed ({stage}): {exc}")
        # GC + memory summary always run so Python-side buffers don't build up
        # across many segments, even when model-unloading is disabled.
        if conf.get("gc_collect", True):
            try:
                import gc
                gc.collect()
            except Exception:  # noqa: BLE001
                pass
        try:
            sm = (self.comfy.memory_summary()
                  if hasattr(self.comfy, "memory_summary") else None)
            if sm:
                print(f"  [mem] after {stage}: {sm}")
        except Exception:  # noqa: BLE001
            pass

    def render_scene(self, scene: dict, scene_idx: int) -> tuple:
        """Render one scene with QC retry loop.

        With ``render.chunked`` enabled, a scene longer than the effective
        segment length is split into several small segments (each at most
        ``min(chunk_seconds, max_segment_seconds)``), rendered as its own tiny
        LTX generation (high FPS + high resolution become feasible because
        every individual generation is small — this prevents OOM at 30-60s
        scenes), then the accepted segments are joined into ``scene_XX.mp4``.
        ``chain_frames`` makes each segment start from the previous segment's
        last frame so the joined shot looks continuous instead of restarted.
        The scene duration comes from ``_effective_scene_seconds()``.

        Returns (video_path, attempts, qc_result, locked_prompts).
        """
        total_seconds = self._effective_scene_seconds()
        chunked = bool(self.render_cfg.get("chunked", False))
        segment_seconds = self._segment_seconds()

        # Single-shot path (chunking disabled or scene fits in one segment).
        if not chunked or total_seconds <= segment_seconds + 1e-6:
            return self._render_segment(scene, scene_idx, duration=total_seconds,
                                        chunk_idx=None, chunk_total=1,
                                        chain_image=None)

        num_chunks = max(1, int(math.ceil(total_seconds / segment_seconds)))
        # BATCH path: submit every segment into ComfyUI's queue up-front so the
        # GPU keeps the same model loaded and renders them back-to-back (no
        # per-segment submit/wait/reload round-trip). Only meaningful when we
        # have several independent segments; frame-chaining needs the previous
        # segment's output so it forces the sequential path instead.
        use_batch = (bool(self.render_cfg.get("batch", True)) and
                     num_chunks > 1 and
                     not bool(self.render_cfg.get("chain_frames", True)))
        if use_batch:
            return self._render_scene_batch(scene, scene_idx, total_seconds,
                                            num_chunks, segment_seconds)

        # ---- sequential chunked path: N segments, joined after creation ----
        fps = int(self.render_cfg.get("fps", 25))
        vw = int(self.render_cfg.get("video_width", 480))
        vh = int(self.render_cfg.get("video_height", 280))
        print(f"\n[chunk] scene {scene_idx} = {total_seconds}s -> "
              f"{num_chunks} x {segment_seconds:g}s segments @ {fps}fps "
              f"({vw}x{vh})")

        chunk_paths: list = []
        chunk_results: list = []
        locked = None
        total_attempts = 0
        chain_image: Optional[str] = None
        for c in range(num_chunks):
            start = c * segment_seconds
            dur = min(segment_seconds, total_seconds - start)
            print(f"\n  [chunk] segment {c + 1}/{num_chunks} "
                  f"({start:.2f}s–{start + dur:.2f}s)")
            path, attempts, qc_result, locked = self._render_segment(
                scene, scene_idx, duration=dur, chunk_idx=c,
                chunk_total=num_chunks, chain_image=chain_image)
            total_attempts += attempts
            passed = qc_result is not None and qc_result.passed
            chunk_results.append({
                "file": os.path.relpath(path, BASE_DIR) if path else "",
                "attempts": attempts,
                "qc": qc_result.metrics if qc_result else {"unchecked": True},
                "passed": passed,
            })
            chunk_paths.append(path)
            # Extract the last frame so the next 1s segment continues it.
            if path and self.render_cfg.get("chain_frames", True):
                chain_image = self._extract_last_frame(path, scene_idx, c)
            # Release leftover RAM/VRAM before the next tiny segment so
            # back-to-back LTX generations do not accumulate memory pressure.
            self._clean_memory("segment")

        scene_path = self._stitch_chunks(scene_idx, chunk_paths)
        passed_count = sum(1 for r in chunk_results if r.get("passed"))
        all_passed = bool(chunk_paths) and passed_count == len(chunk_results)
        agg = QCResult(passed=all_passed, metrics={
            "chunked": True,
            "num_chunks": num_chunks,
            "chunks_passed": passed_count,
            "chunk_seconds": segment_seconds,
            "segments": [r["qc"] for r in chunk_results],
        }, reasons=[] if all_passed else [
            f"{num_chunks - passed_count}/{num_chunks} segments failed QC"])
        return scene_path, total_attempts, agg, locked

    def _render_scene_batch(self, scene: dict, scene_idx: int,
                            total_seconds: float, num_chunks: int,
                            segment_seconds: float) -> tuple:
        """Batch-render a chunked scene by submitting ALL segments into
        ComfyUI's queue up-front, then downloading each as ComfyUI finishes.

        Because the segments are submitted together (and are independent, no
        frame-chaining), ComfyUI keeps the LTX model loaded and processes them
        back-to-back — no per-segment submit/wait/reload round-trip and no
        per-segment model reload, so a tight-memory machine doesn't stall and
        the render is far faster. Memory is only freed once, at scene end.

        Returns (video_path, attempts, qc_result, locked_prompts).
        """
        max_attempts = int(self.qc_cfg.get("max_attempts", 3))
        qc_enabled = bool(self.qc_cfg.get("enabled", True))
        print(f"\n[chunk] BATCH mode: submitting {num_chunks} segments to "
              f"the ComfyUI queue up-front ({segment_seconds:g}s each)")

        # 1) Submit every segment's workflow now, collecting prompt_ids.
        jobs: list[dict] = []
        for c in range(num_chunks):
            start = c * segment_seconds
            dur = min(segment_seconds, total_seconds - start)
            values = self._scene_values(scene, scene_idx, duration=dur,
                                        chunk_idx=c, chunk_total=num_chunks)
            wf = apply_knobs(self.template, self.knobs, values)
            prompt_id = self.comfy.submit(wf)
            print(f"  [chunk] queued segment {c + 1}/{num_chunks} "
                  f"({start:.2f}s–{start + dur:.2f}s) -> {prompt_id}")
            jobs.append({"idx": c, "dur": dur, "pid": prompt_id,
                         "attempts": 1})

        # 2) Wait for + download each in queue order; retry failures.
        chunk_paths: list = []
        chunk_results: list = []
        locked = None
        total_attempts = 0
        timeout = float(self.cfg["comfyui"].get(
            "render_timeout_seconds", 3600))
        if timeout <= 0:
            timeout = 3600
        for job in jobs:
            c = job["idx"]
            dur = job["dur"]
            out_name = f"scene_{scene_idx:02d}_c{c:02d}.mp4"
            accepted = False
            while not accepted and job["attempts"] <= max_attempts:
                pid = job["pid"]
                print(f"  -- segment {c + 1}/{num_chunks} | attempt "
                      f"{job['attempts']}/{max_attempts} (queued {pid})")
                self.comfy.wait(pid, timeout=timeout)
                src = self._find_output(pid)
                if src is None:
                    print("  !! no output video found in ComfyUI history")
                    job["pid"] = self.comfy.submit(
                        apply_knobs(self.template, self.knobs,
                                    self._scene_values(
                                        scene, scene_idx, duration=dur,
                                        chunk_idx=c, chunk_total=num_chunks)))
                    job["attempts"] += 1
                    continue
                dest = os.path.join(self.output_dir, out_name)
                self.comfy.download(src["filename"], dest,
                                    src.get("subfolder", ""),
                                    src.get("type", "output"))
                print(f"  downloaded -> "
                      f"{os.path.relpath(dest, BASE_DIR)}")
                attempts = job["attempts"]
                total_attempts += attempts
                if not qc_enabled:
                    chunk_paths.append(dest)
                    chunk_results.append({
                        "file": os.path.relpath(dest, BASE_DIR),
                        "attempts": attempts,
                        "qc": {"unchecked": True}, "passed": True})
                    accepted = True
                    break
                expected_frames = int(round(
                    dur * int(self.render_cfg.get("fps", 25))))
                min_frames = max(2, expected_frames - 2)
                min_dur = max(0.3, dur * 0.5)
                result = self.qc.analyze(dest, dur, min_frames=min_frames,
                                         min_duration_seconds=min_dur)
                print(f"  QC: {'PASS' if result.passed else 'FAIL'} "
                      f"{result.summary()}")
                if result.passed:
                    chunk_paths.append(dest)
                    chunk_results.append({
                        "file": os.path.relpath(dest, BASE_DIR),
                        "attempts": attempts, "qc": result.metrics,
                        "passed": True})
                    accepted = True
                    break
                # QC failed -> re-submit this segment with fresh seeds.
                print("  QC FAILED: " + "; ".join(result.reasons) +
                      " - re-shooting with new seeds")
                values = self._scene_values(scene, scene_idx, duration=dur,
                                            chunk_idx=c,
                                            chunk_total=num_chunks)
                job["pid"] = self.comfy.submit(
                    apply_knobs(self.template, self.knobs, values))
                job["attempts"] += 1
            if not accepted:
                # exhausted attempts: keep the last file if it exists
                chunk_paths.append("")
                chunk_results.append({
                    "file": "", "attempts": job["attempts"],
                    "qc": {"unchecked": True}, "passed": False})

        # 3) Release leftover RAM/VRAM once, after the whole scene's queue.
        self._clean_memory("scene")

        scene_path = self._stitch_chunks(scene_idx, chunk_paths
                                         if chunk_paths else [])
        passed_count = sum(1 for r in chunk_results if r.get("passed"))
        all_passed = bool(chunk_paths) and passed_count == len(chunk_results)
        agg = QCResult(passed=all_passed, metrics={
            "chunked": True, "batch": True, "num_chunks": num_chunks,
            "chunks_passed": passed_count,
            "chunk_seconds": segment_seconds,
            "segments": [r["qc"] for r in chunk_results],
        }, reasons=[] if all_passed else [
            f"{num_chunks - passed_count}/{num_chunks} segments failed QC"])
        return scene_path, total_attempts, agg, locked

    def _render_segment(self, scene: dict, scene_idx: int, duration: float,
                        chunk_idx: Optional[int], chunk_total: int,
                        chain_image: Optional[str]) -> tuple:
        """Render ONE clip (a full scene, or one 1s segment of a chunked scene)
        with the QC retry loop. ``chain_image`` (a PNG path) replaces the T2I
        keyframe so a segment continues the previous segment's last frame.
        Returns (video_path, attempts, qc_result, locked_prompts)."""
        max_attempts = int(self.qc_cfg.get("max_attempts", 3))
        qc_enabled = bool(self.qc_cfg.get("enabled", True))
        out_name = (f"scene_{scene_idx:02d}.mp4"
                    if chunk_idx is None
                    else f"scene_{scene_idx:02d}_c{chunk_idx:02d}.mp4")
        locked_prompts = None

        for attempt in range(1, max_attempts + 1):
            values = self._scene_values(scene, scene_idx, duration=duration,
                                        chunk_idx=chunk_idx,
                                        chunk_total=chunk_total)
            locked_prompts = (values["image_prompt"], values["video_prompt"])
            wf = apply_knobs(self.template, self.knobs, values)
            if chain_image and chunk_idx is not None and chunk_idx > 0:
                wf = self._apply_chain_image(wf, chain_image)

            if chunk_idx is None:
                print(f"\n--- scene {scene_idx} | attempt "
                      f"{attempt}/{max_attempts} ---")
            else:
                print(f"  -- segment {chunk_idx + 1}/{chunk_total} | "
                      f"attempt {attempt}/{max_attempts}")
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
                return dest, attempt, None, locked_prompts
            # 1s segments are short (e.g. 25-61 frames): scale the minimum
            # frame count to the segment so a short clip isn't rejected by a
            # threshold tuned for ~5s clips.
            expected_frames = int(round(
                duration * int(self.render_cfg.get("fps", 25))))
            min_frames = max(2, expected_frames - 2)
            # Short segments must not be rejected by a min-duration threshold
            # tuned for long clips; scale it to the segment length.
            min_dur = max(0.3, duration * 0.5)
            result = self.qc.analyze(dest, duration, min_frames=min_frames,
                                     min_duration_seconds=min_dur)
            print(f"  QC: {'PASS' if result.passed else 'FAIL'}")
            print(f"       {result.summary()}")
            if result.passed:
                return dest, attempt, result, locked_prompts
            print("  QC FAILED: " + "; ".join(result.reasons) +
                  " - re-shooting with new seeds")
        # exhausted attempts: keep last file, record failure
        last = os.path.join(self.output_dir, out_name)
        if os.path.exists(last):
            return last, max_attempts, None, locked_prompts
        return "", max_attempts, None, locked_prompts

    def _apply_chain_image(self, wf: dict, image_path: str) -> dict:
        """Swap a segment's first frame for the previous segment's last frame
        (I2V chaining). Uploads the PNG to ComfyUI, adds a LoadImage node and
        repoints the video first-frame chain (node 26) at it, then drops the
        now-unused T2I keyframe subgraph so continuation segments skip image
        generation. On any failure the workflow is returned unchanged and the
        segment falls back to the normal keyframe."""
        try:
            up = self.comfy.upload_image(image_path)
            name = up.get("name") or os.path.basename(image_path)
            sub = up.get("subfolder", "") or ""
        except Exception as exc:  # noqa: BLE001
            print(f"  [chunk] !! could not upload chained frame: {exc}")
            return wf
        load_name = f"{sub}/{name}" if sub else name
        used = {int(k) for k in wf if str(k).lstrip('-').isdigit()}
        nid = (max(used) + 1) if used else 60
        wf[str(nid)] = {
            "class_type": "LoadImage",
            "inputs": {"image": load_name, "upload": "image"},
        }
        # The video first-frame chain: node 26 (ResizeImageMaskNode) consumes
        # the keyframe (node 11). Point it at the uploaded last frame instead.
        if "26" in wf and "input" in wf["26"].get("inputs", {}):
            wf["26"]["inputs"]["input"] = [str(nid), 0]
            # The T2I keyframe subgraph (nodes 1..11) now feeds nothing.
            for k in [str(i) for i in range(1, 12)]:
                wf.pop(k, None)
        return wf

    def _extract_last_frame(self, video_path: str, scene_idx: int,
                            chunk_idx: int) -> Optional[str]:
        """Extract the exact last frame of a rendered chunk to a PNG so the
        next chunk can start from it (continuous motion across segments)."""
        png_dir = os.path.join(self.output_dir, "_chunk_frames")
        os.makedirs(png_dir, exist_ok=True)
        png = os.path.join(png_dir,
                           f"scene_{scene_idx:02d}_c{chunk_idx:02d}_last.png")
        try:
            import imageio.v2 as imageio
            last = None
            with imageio.get_reader(video_path, "ffmpeg") as rdr:
                for frame in rdr:
                    last = frame
            if last is None:
                return None
            imageio.imwrite(png, last)
            return png
        except Exception as exc:  # noqa: BLE001
            print(f"  [chunk] !! could not extract last frame: {exc}")
            return None

    def _stitch_chunks(self, scene_idx: int, chunk_paths: list) -> str:
        """Join a scene's 1s segments into the scene file (scene_XX.mp4)."""
        out = os.path.join(self.output_dir, f"scene_{scene_idx:02d}.mp4")
        existing = [p for p in chunk_paths if p and os.path.exists(p)]
        if not existing:
            print(f"  [chunk] !! no valid segments to stitch for scene "
                  f"{scene_idx}")
            return ""
        if len(existing) == 1:
            shutil.copyfile(existing[0], out)
            print(f"  [chunk] single segment -> "
                  f"{os.path.relpath(out, BASE_DIR)}")
            return out
        print(f"  [chunk] stitching {len(existing)} segments -> "
              f"{os.path.relpath(out, BASE_DIR)}")
        return self._ffmpeg_concat(existing, out)

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
    def _ffmpeg_concat(self, paths: list, out: str) -> str:
        """Concatenate video files with ffmpeg's concat demuxer (audio dropped
        so streams of independent clips never clash). Returns out on success."""
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:  # noqa: BLE001
            ffmpeg = "ffmpeg"
        list_file = os.path.join(self.output_dir, "_concat.txt")
        with open(list_file, "w", encoding="utf-8") as f:
            for p in paths:
                f.write(f"file '{p.replace(chr(39), chr(39)*4)}'\n")
        fps = int(self.render_cfg.get("fps", 25))
        cmd = [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", list_file,
               "-r", str(fps), "-c:v", "libx264", "-crf", "18",
               "-pix_fmt", "yuv420p", "-an", out]
        print("  " + " ".join(cmd))
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=1200)
            return out
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"!! stitch failed: {e}")
            return ""

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
        out = os.path.join(self.output_dir, film_name)
        print("\n== stitching film ==")
        return self._ffmpeg_concat(resolved, out)

    def _select_scenes(self, scenes: list, only_scene: Optional[int] = None,
                       scene_count: Optional[int] = None) -> tuple[list, int]:
        """Return the subset of scenes to render and the base scene index."""
        if only_scene is not None:
            if not (0 <= only_scene < len(scenes)):
                raise ValueError(f"scene index {only_scene} out of range (0..{len(scenes)-1})")
            return [scenes[only_scene]], only_scene
        if scene_count is not None:
            if scene_count <= 0:
                raise ValueError("scene_count must be positive")
            count = min(int(scene_count), len(scenes))
            return scenes[:count], 0
        return scenes, 0

    # ------------------------------------------------------------- main run
    def run(self, only_scene: Optional[int] = None,
            scene_count: Optional[int] = None) -> None:
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

        # Persist the generated story (LLM or custom) so the dashboard's
        # Story Writer tab can display and edit it between runs.
        try:
            story_doc = {
                "story_title": title,
                "scenes": [
                    {
                        "index": i + 1,
                        "image_prompt": s.get("image_prompt", ""),
                        "video_prompt": s.get("video_prompt", ""),
                        "characters_present": list(s.get("characters_present", [])),
                        "dialogue": s.get("dialogue") or "",
                        "audio_lines": s.get("audio_lines", ""),
                    }
                    for i, s in enumerate(scenes)
                ],
            }
            story_path = os.path.join(self.output_dir, "story.json")
            with open(story_path, "w", encoding="utf-8") as f:
                json.dump(story_doc, f, ensure_ascii=False, indent=2)
        except Exception as exc:  # noqa: BLE001
            print(f"[story] could not persist story.json: {exc}")

        # Character locking (Phase 0/1/2): extract cast, write descriptors,
        # render master references (cached per character, reused by name).
        if not self.cfg.get("characters", {}).get("enabled", True):
            print("[char] character locking disabled (characters.enabled=false)")
        else:
            try:
                self.chars.ensure_locks(
                    story,
                    generate_refs=bool(self.cfg.get("characters", {}).get(
                        "auto_generate_refs", True)),
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[char] library setup failed, continuing with "
                      f"prompt-only: {exc}")

        try:
            scenes, base_idx = self._select_scenes(scenes, only_scene=only_scene,
                                                   scene_count=scene_count)
        except ValueError as exc:
            print(f"!! {exc}")
            return

        results = []
        manual_review = bool(self.cfg["director"].get("manual_review", False))
        for i, scene in enumerate(scenes):
            idx = base_idx + i
            print(f"\n{'='*60}\nSCENE {idx+1}: {scene['video_prompt'][:60]}…\n"
                  f"{'='*60}")
            path, attempts, qc_result, locked = self.render_scene(scene, idx)
            passed = qc_result is not None and qc_result.passed
            locked_img, locked_vid = (locked if locked
                                      else (scene["image_prompt"],
                                            scene["video_prompt"]))
            results.append({
                "scene": idx + 1,
                "attempts": attempts,
                "qc_passed": passed,
                "qc": qc_result.metrics if qc_result else {"unchecked": True},
                "file": os.path.relpath(path, BASE_DIR) if path else "",
                "image_prompt": locked_img,
                "video_prompt": locked_vid,
                "source_image_prompt": scene["image_prompt"],
                "characters_present": scene.get("characters_present", []),
                "dialogue": scene.get("dialogue"),
                "audio_lines": scene.get("audio_lines", ""),
            })
            if manual_review and passed:
                input(f"  Human review: press Enter to accept scene {idx+1} "
                      "or Ctrl+C to abort…")
            # Unload cached models between scenes: the biggest single source of
            # RAM/VRAM pressure across a whole run (LTX fp8 + 12B text
            # encoder staying resident in ComfyUI).
            self._clean_memory("scene")

        report = {
            "story_title": title,
            "config": self.cfg,
            "characters": self.chars.summary(),
            "scenes": results,
        }
        report_path = os.path.join(self.output_dir, "report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        if only_scene is None and scene_count is None:
            film = self.stitch([r["file"] for r in results])
            report["final_film"] = os.path.relpath(film, BASE_DIR) if film else ""
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(f"\n== film saved -> {report.get('final_film', 'N/A')}")

            # Phase 4b: narration - speak each scene's DIALOGUE as audio and
            # burn subtitles into the film. Only active when audio.enabled;
            # purely additive and never breaks the base film on failure.
            if self.cfg.get("audio", {}).get("enabled", False):
                try:
                    import narrate as narrate_mod
                    narrated = narrate_mod.narrate(self.cfg, report,
                                                   self.output_dir)
                    if narrated:
                        report["final_film_narrated"] = os.path.relpath(
                            narrated, BASE_DIR)
                        with open(report_path, "w", encoding="utf-8") as f:
                            json.dump(report, f, ensure_ascii=False, indent=2)
                except Exception as exc:  # noqa: BLE001
                    print(f"[narrate] auto-narration skipped: {exc}")
        else:
            print(f"\n== {'scene' if only_scene is not None else 'scene range'} rendered -> {results[0]['file'] if results else 'N/A'}")

        # Phase 4: cross-scene consistency review
        if self.chars.enabled:
            try:
                self.chars.write_consistency_report(story, results,
                                                    self.output_dir)
            except Exception as exc:  # noqa: BLE001
                print(f"[char] consistency report failed: {exc}")

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
                        help="render one scene index (0-based)")
    parser.add_argument("--scene-count", type=int, default=None,
                        help="render this many scenes starting at --scene")
    parser.add_argument("--no-characters", action="store_true",
                        help="disable character locking for this run")
    parser.add_argument("--no-chunks", action="store_true",
                        help="disable 1-second chunked rendering (render each "
                             "scene as one long clip instead)")
    args = parser.parse_args(argv)

    config = load_json(args.config)
    if args.no_characters:
        config.setdefault("characters", {})["enabled"] = False
    if args.no_chunks:
        config.setdefault("render", {})["chunked"] = False
    log_path = _install_log_tee(config)
    print(f"== log -> {os.path.relpath(log_path, BASE_DIR)}")
    director = Director(config)
    if args.check:
        problems = director.check_environment()
        print("\n" + ("ALL CHECKS PASSED" if not problems
                      else "\n".join("- " + p for p in problems)))
        return
    director.run(only_scene=args.scene, scene_count=args.scene_count)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nDirector stopped by user.")
        sys.exit(130)
    except ComfyUIError as e:
        print(f"\nDIRECTOR ERROR: {e}")
        sys.exit(1)
