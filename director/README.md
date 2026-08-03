# ComfyUI Director Mode — Story → Image → Video with automatic QC

A "director model" that drives your existing **ComfyUI** workflow end-to-end:

> **Story writer** (optional LLM, or your own text) → **Text-to-Image** (Z-Image-Turbo) → **Image-to-Video** (LTX-2.3) → **1-second clips (chunked)** → **automatic QC pass/fail** → **re-shoot on failure** → **join segments → next scene → join scenes → final film**.

The Director app is plain Python. It talks to ComfyUI's HTTP API, so **no node-graph can do the QC loop — this is why the director is a Python app driving ComfyUI** (industry-standard approach).

---

## Folder contents

| File | Purpose |
|------|---------|
| `build_workflow.py` | Flattens `Text to Image and Image to Video.json` into an API-format scene graph. Re-run after editing the source workflow. |
| `workflow_scene_template.json` | **Generated** — one 50-node scene graph (T2I → first frame → I2V → SaveVideo). |
| `workflow_knobs.json` | **Generated** — which template nodes are the "knobs" (prompts, seeds, save path). |
| `config.json` | All settings (ComfyUI URL, LLM, scene length, QC thresholds). |
| `storywriter.py` | Story generation: custom story bypass → LLM → template fallback. |
| `comfy_api.py` | Minimal ComfyUI HTTP client (submit / poll / download). |
| `qc.py` | Quality Control: duration, black frames, frozen frames, motion. |
| `director.py` | The Director orchestrator (main entry point). |
| `narrate.py` | Optional edge-tts narration: speaks each scene's `dialogue` as audio + burns subtitles (auto-runs after the film when `audio.enabled`). |
| `custom_story.txt` | Optional: put your own story here to bypass the LLM. |
| `requirements.txt` | Python dependencies. |
| `output/` | Rendered clips + `report.json` + stitched film. |
| `tests/` | A mock ComfyUI server + configs to test the full pipeline **without a GPU** (see below). |

---

## Requirements

- **ComfyUI Desktop** running with the API enabled (default `http://127.0.0.1:8188`).
- The same models your workflow uses (already present if the workflow runs):
  - T2I: `z_image_turbo_int8_convrot.safetensors`, `qwen_3_4b_fp8_mixed.safetensors`, `ae.safetensors`
  - I2V: `ltx-2.3-22b-dev-fp8.safetensors`, `gemma_3_12B_it_fp4_mixed_2.safetensors`, `ltx-2.3-spatial-upscaler-x2-1.1.safetensors`, distilled LoRA
- Python 3.10+ with:
  ```
  pip install -r requirements.txt
  ```
- Optional for the AI story writer: **Ollama** (`ollama pull gemma3:12b`) or any OpenAI-compatible server (LM Studio, vLLM, etc.). **Not needed if you use a custom story or `"backend": "none"`.**

---

## Quick start

```bash
cd director
python director.py --check      # verify ComfyUI is up + all models exist (no render)
python director.py              # full production run
```

The director will:
1. Print an environment/model check.
2. Get the story (see below).
3. Shoot each scene — by default split into **1-second segments** rendered at
   the configured FPS/resolution (e.g. 1080×1920 @ 30fps, 9:16 vertical), then
   joined right after creation into `scene_XX.mp4` (see
   ["1-second chunked rendering"](#1-second-chunked-rendering-high-fps--high-resolution)).
4. Run QC on every clip/segment. **Fail → automatic re-shoot with new seeds** (up to `qc.max_attempts`).
5. Stitch the accepted scene clips into `output/final_film.mp4`.
6. Write `output/report.json` (per-scene results + QC metrics, including the
   per-segment metrics when chunked).

### One scene only

```bash
python director.py --scene 2    # render only scene index 2 (0-based)
```

---

## 1-second chunked rendering (high FPS + high resolution)

Long scenes at high FPS/high resolution are rendered as **one giant LTX
generation** (`seconds × fps + 1` frames) which can OOM the GPU, hang, or run
out of memory on the RTX 4080. The director avoids that by splitting every
scene into **small segments** (at most `render.max_segment_seconds`), rendering
each segment as its own tiny generation, then **joining the segments right
after creation** — and finally joining the scenes. A **hard safety cap** means
a single generation is never huge no matter what `chunk_seconds` says, so
30–60 s videos render reliably.

```jsonc
"render": {
    "video_width": 1080,      // 9:16 vertical (Reels / TikTok / Shorts)
    "video_height": 1920,
    "fps": 30,
    "image_width": 1080,      // keyframe T2I matches the portrait frame
    "image_height": 1920,
    "chunked": true,          // split each scene into small segments
    "chunk_seconds": 1.0,     // preferred segment length (s)
    "max_segment_seconds": 1.0, // HARD CAP — one generation never exceeds this
    "chain_frames": true      // continue each segment from the previous
                              // segment's last frame (continuous shot)
}
```

How it works (`director.py`):

- A `N`-second scene becomes `ceil(N / segment)` segments where
  `segment = min(chunk_seconds, max_segment_seconds)` — the cap wins, so even
  if `chunk_seconds` is set to 30 the pipeline still submits 1 s generations
  (e.g. 1s @ 30fps = 31 frames — tiny, fast, VRAM-safe) instead of one 751-frame
  generation that OOMs the GPU.
- **Total video length:** set `story.video_length` (the whole video's runtime)
  and each scene is auto-adjusted to `video_length / scenes`, so "make it 60 s"
  with 12 scenes gives 5 s per scene, each rendered as 5 × 1 s chunks. Leave it
  at 0 (or unset) to use `story.seconds_per_scene` directly.
- Each segment is QC'd independently with the same pass/fail + re-shoot loop.
- With `chain_frames: true`, the **last frame** of segment `k` is extracted to
  a PNG, uploaded to ComfyUI (`/upload/image`), and injected as the **first
  frame** of segment `k+1` (the T2I keyframe subgraph is skipped for
  continuation segments) — so the joined shot is a continuous camera move, not
  a series of restarts.
- All accepted segments are joined with ffmpeg into `scene_XX.mp4`, then all
  scenes are joined into `final_film.mp4`.
- Disable it for a classic single-shot render: `"chunked": false` or run
  `python director.py --no-chunks`.

> **Note on 1080×1920:** LTX latent dims in this template are the video
> dimensions halved (`ComfyMathExpression "a/2"`), so the latent is
> 540×960. If your ComfyUI/LTX build rejects that resolution, pick video
> dimensions divisible by 64 (e.g. **1088×1920** — still ~9:16) in
> `render.video_width` / `render.video_height`.

---

## Story source — 3 ways (custom story is the optional bypass)

Priority:

1. **Custom story (bypass, NO LLM)** — fill in `custom_story.txt` (see the file header
   for the `IMAGE:` / `VIDEO:` / `---` format). If the file has any content, the
   Director uses it and never calls the LLM. **This is your "text box for custom
   story" requirement — an on-disk text box.**
2. **LLM story** — with `config.json` → `"llm": {"backend": "ollama"}` (default)
   or `"openai"`, the director asks the model for a JSON story with
   `story.scenes` scenes. Set the model/URL in `config.json`.
3. **Template fallback** — if no custom story and no LLM, a small built-in story
   runs so the pipeline is always testable.

Control length via `story.scenes`, `story.seconds_per_scene`, or
`story.video_length` (total runtime, divided across scenes) in `config.json`
(or the dashboard — Story & LLM tab). For a 30–60 s video, prefer
`story.video_length`; the per-scene duration is computed automatically and
every scene is still rendered as small 1 s chunks, so it never OOMs.

---

## QC (pass/fail) — what the director checks

Configured under `qc` in `config.json`:

| Check | Meaning | Default |
|-------|---------|---------|
| `duration_tolerance_seconds` | clip length vs requested (truncated export) | 0.6 |
| `min_duration_seconds` | clip too short | 4.0 |
| `max_black_frames_pct` | all-black frames → empty generation | 15% |
| `max_frozen_frames_pct` | identical consecutive frames → stall | 25% |
| `min_motion_std` | too static / no motion | 2.0 |
| `min_frames` | too few frames | 40 |
| `max_attempts` | re-shoots before giving up | 3 |

Set `"qc": {"enabled": false}` to skip QC. Set
`"director": {"manual_review": true}` to pause for a human between scenes.

---

## How the scene graph was built (`build_workflow.py`)

Your `Text to Image and Image to Video.json` uses ComfyUI Desktop **subgraphs**.
ComfyUI's `/prompt` API only accepts a flat graph, so `build_workflow.py`
flattens the two subgraphs into one scene graph:

- T2I subgraph → Z-Image-Turbo (prompt, seed, steps) → VAEDecode
- first frame → `ImageScaleToMaxDimension` (1080) → `ImageFromBatch`
- I2V subgraph → LTX-2.3 base pass (seed from `video_seed`) → latent ×2 upscale
  → refine pass → decode → `CreateVideo` → `SaveVideo`
- The LTX **prompt-enhance path** (Gemma LLM inside the workflow +
  `ComfySwitchNode`) is **bypassed**: the positive encoder gets the raw scene
  prompt directly, so the director's text is used verbatim.
- Dead nodes (`TextGenerateLTX2Prompt`, `PreviewAny`, enhance-only `LoraLoader`)
  are dropped.

Knobs (things the director changes per scene) live in `workflow_knobs.json`:
`image_prompt`, `image_seed`, `video_prompt`, `video_seed`, `save_prefix`.

> **Manual fallback (30s):** if you ever change the source workflow, you can
> regenerate the template with `python build_workflow.py`, OR in ComfyUI Desktop
> open the workflow → **Save (API Format)** → drop that JSON in as
> `workflow_scene_template.json` and adjust `workflow_knobs.json` to point at
> the CLIPTextEncode (positive) text inputs, the KSampler/RandomNoise seeds, and
> SaveVideo filename_prefix.

---

## Notes / limitations

- With `render.chunked: true` (default), every scene is rendered as short
  segments (hard-capped at `render.max_segment_seconds`, default **1.0 s**)
  that are joined right after creation; total render time is similar to a
  single long render but each individual generation is small, so high FPS +
  high resolution never OOM/hang (expect roughly 1–5 min per 1s segment on a
  mid-range GPU). The cap is enforced in code even if `chunk_seconds` is set
  higher, so long 30/60 s scenes can never OOM an 8–16 GB GPU.
- Without chunking, one ComfyUI prompt runs the whole image+video scene; expect
  roughly 1–5 min per 5s scene on a mid-range GPU — long durations/high FPS
  risk GPU OOM.
- **RAM / VRAM cleanup:** between scenes the director asks ComfyUI to unload
  model weights and free cached memory via `POST /free` (so the LTX fp8 model +
  12B text encoder do not stay resident and eat RAM/VRAM across a whole run)
  and forces Python GC to drop downloaded-file buffers. Tune with
  `comfyui.free_between_scenes` (default `true`) and
  `comfyui.free_between_segments` (default `false` — freeing between every 1s
  segment forces the I2V model to reload each time, which is slower; enable it
  only if a single scene still OOMs). Each cleanup prints the live
  `VRAM/RAM free` figures so you can watch pressure in the console. If you
  still see high RAM/VRAM with chunked 1s segments, also run ComfyUI Desktop in
  `--lowvram` mode (Settings → ComfyUI → extra launch args).
- The final stitch re-encodes clips (H.264) to guarantee concatenation works;
  audio from LTX is currently dropped in the stitch (`-an`) for robustness.
- **Dialogue → audio:** each scene's `dialogue` (from `custom_story.txt`
  `DIALOGUE:` / `Speaker: line` blocks, or the LLM story writer) is spoken by
  `narrate.py` with `edge-tts` (per-character voice locks from `characters/`
  when set, else the global narrator voice) and subtitles are burned in. This
  runs **automatically** at the end of `director.py` when `audio.enabled: true`
  and produces `final_film_narrated.mp4` (the base film is never modified). A
  defensive guard ensures the story genre/prompt is **never** read out loud,
  and `audio_lines` (ambient sound cues) are never spoken.
- The LLM is only a **story writer**; all images/videos are generated locally by
  ComfyUI. No cloud API keys are required.
- Change the film look by editing prompts, `config.json` (resolution, fps,
  steps), or the source workflow and re-running `build_workflow.py`.

---

## Companion interactive workflow

Next to `director/` is **`Story to Video - Director (T2I to I2V).json`** — a
manual ComfyUI Desktop workflow (also derived from your original). If you prefer
to run scenes by hand instead of the Python director, open it in ComfyUI Desktop
and just fill in the **STORY**, **CHARACTER** and **VIDEO SCRIPT** text boxes.
It has no automatic QC loop (a node graph cannot branch/re-run), so it is the
manual alternative to `director.py`.

The two are independent: edit either and the other is unaffected.

---

## Testing the pipeline WITHOUT a GPU (mock ComfyUI)

`tests/mock_comfyui.py` is a tiny fake ComfyUI server. It accepts real
API-format workflows, "renders" synthetic 5s clips, and — on purpose — makes
the **1st attempt of every scene a broken (black/static) video** so the
director's automatic re-shoot loop is exercised, then succeeds. This verifies
the entire pipeline (story → knobs → submit → download → QC → retry → stitch →
report) with no GPU and no real ComfyUI.

```bash
# terminal 1 — start the mock
python tests/mock_comfyui.py --port 8899

# terminal 2 — full run with a 2-scene custom story (tests the bypass too)
python director.py --config tests\config_mock_custom.json

# full run with no custom story / no LLM -> built-in fallback story
python director.py --config tests\config_mock_fallback.json

# chunked e2e — 3 scenes x 2s as 1s segments @ 30fps (1080x1920 9:16),
# with frame chaining + per-scene join (tests config_mock_chunk.json)
python director.py --config tests\config_mock_chunk.json
```

Expected result: every scene shows `QC: FAIL` on attempt 1 then `QC: PASS` on
attempt 2, and a valid `tests/output_test/final_film.mp4` (4 clips × 5s = 20s)
is produced. This is exactly what the real ComfyUI run does, just with a fake
renderer. The chunked run (`config_mock_chunk.json`) produces 1s segment files
(`scene_XX_cYY.mp4`), joined `scene_XX.mp4` files and a 6s `final_film.mp4`.

> The mock also made the model check (`--check`) honest: it serves a realistic
> `/object_info` (the `[choices, options]` shape real ComfyUI returns), which is
> what caught and fixed a bug in the original model-verification code.


