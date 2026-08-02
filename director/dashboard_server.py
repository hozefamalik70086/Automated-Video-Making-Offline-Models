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
    "launching": False,  # atomic guard: a run is being spawned (race-safe)
}
_run_lock = threading.Lock()

# ------------------------------------------------- story completion state
# Background LLM task that fills missing scene details (distinct motion
# video prompts + ambient audio lines) and writes them back to the story.
_complete = {
    "thread": None,
    "state": "idle",   # idle | running | done | cancelled
    "mode": "",        # fill | generate
    "total": 0,
    "done": 0,
    "filled": 0,
    "errors": [],
    "model": "",
    "warning": "",
    "current": "",
    "started": None,
    "cancel": False,   # set by /api/story/cancel; worker stops between scenes
}
_complete_lock = threading.Lock()


def _cancel_requested() -> bool:
    """True if the user asked to cancel the running story task."""
    with _complete_lock:
        return _complete["cancel"]


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


def _load_story() -> dict:
    """Read the last generated story (LLM or custom) from output/story.json,
    falling back to parsing the configured custom_story_file."""
    rp = os.path.join(BASE_DIR, _out_dir(), "story.json")
    try:
        with open(rp, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        pass
    fname = str(_read_config().get("story", {}).get(
        "custom_story_file", "custom_story.txt"))
    return _parse_custom_story_file(fname)


def _parse_custom_story_file(fname: str) -> dict:
    """Minimal parser for custom_story.txt -> {story_title, scenes} so the
    Story Writer can show the current story before a director run persists
    story.json."""
    import re as _re
    path = os.path.join(BASE_DIR, fname)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except Exception:
        return {}
    text = "\n".join(ln for ln in text.splitlines()
                     if not ln.strip().startswith("#")).strip()
    blocks = [b.strip() for b in _re.split(r"^\s*-{3,}\s*$", text,
                                           flags=_re.M) if b.strip()]
    if not blocks:
        return {}
    scenes = []
    for idx, block in enumerate(blocks, start=1):
        img = vid = chars = dia = ""
        audio = []
        rest = []
        for ln in block.splitlines():
            low = ln.strip().lower()
            if low.startswith("image:"):
                img = ln.split(":", 1)[1].strip()
            elif low.startswith("video:"):
                vid = ln.split(":", 1)[1].strip()
            elif low.startswith("characters:"):
                chars = ln.split(":", 1)[1].strip()
            elif low.startswith("dialogue:"):
                dia = ln.split(":", 1)[1].strip()
            elif low.startswith("voice:"):
                audio.append(ln.split(":", 1)[1].strip())
            else:
                rest.append(ln.strip())
        body = " ".join(rest).strip()
        if not img:
            img = body
        if not vid:
            vid = body
        if not (img or vid):
            continue
        scenes.append({
            "index": idx,
            "image_prompt": img,
            "video_prompt": vid,
            "characters_present": [x.strip() for x in chars.split(",")
                                   if x.strip()] if chars else [],
            "dialogue": dia or None,
            "audio_lines": "\n".join(audio),
        })
    if not scenes:
        return {}
    return {"story_title": "Custom story", "scenes": scenes}


def _story_to_custom_text(story: dict) -> str:
    """Convert story.json scene dicts back to editable custom_story.txt format."""
    out = ["# Story Writer export — edit freely, the Director uses this file.",
           "# One block per scene, separated by ---."]
    for i, sc in enumerate((story.get("scenes") or []), 1):
        out.append(f"# Scene {i}")
        img = str(sc.get("image_prompt") or "").strip()
        if img:
            out.append("IMAGE: " + img)
        vid = str(sc.get("video_prompt") or "").strip()
        if vid:
            out.append("VIDEO: " + vid)
        chars = sc.get("characters_present") or []
        if chars:
            out.append("CHARACTERS: " + ", ".join(str(c).strip() for c in chars))
        dia = str(sc.get("dialogue") or "").strip()
        if dia:
            out.append("DIALOGUE: " + dia)
        audio = sc.get("audio_lines") or ""
        if isinstance(audio, (list, tuple)):
            audio = " ".join(str(a) for a in audio)
        audio = str(audio).strip()
        if audio:
            out.append("VOICE: " + audio)
        out.append("---")
    return "\n".join(out) + "\n"


def _write_story(story: dict) -> None:
    """Persist the story to output/story.json and export an editable custom
    story file so the next run uses the edited version (LLM bypassed)."""
    rp = os.path.join(BASE_DIR, _out_dir(), "story.json")
    with open(rp, "w", encoding="utf-8") as f:
        json.dump(story, f, ensure_ascii=False, indent=2)
    fname = str(_read_config().get("story", {}).get(
        "custom_story_file", "custom_story.txt")).replace("..", "")
    with open(os.path.join(BASE_DIR, fname), "w", encoding="utf-8") as f:
        f.write(_story_to_custom_text(story))


# ------------------------------------------------- LLM story completion
def _ollama_models() -> list:
    """Names of models currently loaded on the Ollama server."""
    import requests
    try:
        llm = _read_config().get("llm", {})
        url = str(llm.get("ollama_url",
                          "http://127.0.0.1:11434")).rstrip("/")
        r = requests.get(url + "/api/tags", timeout=5)
        r.raise_for_status()
        return [m.get("name", "") for m in r.json().get("models", [])]
    except Exception:
        return []


def _model_key(name: str) -> str:
    """'gemma4' and 'gemma4:latest' both key to 'gemma4', so a plain model
    name (no tag) matches the same model /api/tags reports with ':latest'."""
    name = (name or "").strip()
    if ":" in name:
        base, tag = name.rsplit(":", 1)
        if tag in ("latest", ""):
            return base.strip()
    return name


def _pick_llm_model() -> tuple:
    """Return (model_name, warning). Prefer the configured model (matching by
    name even if the tag differs, e.g. 'gemma4' == 'gemma4:latest'), but fall
    back to a known-working non-thinking model if the configured one is not
    available on this Ollama install (e.g. qwen3:4b may be missing)."""
    llm = _read_config().get("llm", {})
    configured = str(llm.get("ollama_model", "") or "").strip()
    available = _ollama_models()
    if configured:
        ck = _model_key(configured)
        for name in available:
            if name == configured or _model_key(name) == ck:
                return name, ""   # use Ollama's canonical name (gemma4:latest)
    # non-thinking models work reliably for this JSON task
    prefer = ["gemma4:latest", "qwen3.5:latest", "qwen3.6:latest",
              "qwen3-vl:4b", "qwen3-vl:2b", "ornith:9b"]
    for m in prefer:
        if m in available:
            warn = (f"configured model '{configured or '(none)'}' is not "
                    f"available on Ollama; using '{m}' instead. Update "
                    f"llm.ollama_model in config.json to '{m}'.")
            return m, warn
    if configured:
        return configured, (f"configured model '{configured}' is not in "
                            f"Ollama's model list; it will likely fail. Pull "
                            f"it with 'ollama pull {configured}' or change "
                            f"llm.ollama_model.")
    return "", "no Ollama model configured (set llm.ollama_model)"


def _llm_chat(messages, model: str, json_format: bool = False) -> str:
    """One non-streaming Ollama chat call -> assistant content string."""
    import requests
    llm = _read_config().get("llm", {})
    url = str(llm.get("ollama_url", "http://127.0.0.1:11434")).rstrip("/")
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": float(llm.get("temperature", 0.7)),
            "num_predict": int(llm.get("max_tokens", 1024)),
        },
    }
    if json_format:
        payload["format"] = "json"
    resp = requests.post(url + "/api/chat", json=payload, timeout=240)
    resp.raise_for_status()
    data = resp.json()
    return (data.get("message") or {}).get("content", "") or ""


def _extract_json_string(content: str, key: str) -> str:
    """Pull `key: "value"` out of a (possibly JSON-wrapped / markdown) reply."""
    import re as _re
    content = content.strip()
    m = _re.search(r'"' + _re.escape(key) + r'"\s*:\s*'
                   r'"((?:[^"\\]|\\.)*)"', content, _re.S)
    if m:
        raw = m.group(1)
        try:
            return json.loads('"' + raw + '"')
        except Exception:
            return raw.replace("\\n", " ").strip()
    try:  # whole reply is a JSON object
        obj = json.loads(content)
        if isinstance(obj, dict):
            v = obj.get(key)
            if v:
                return str(v).strip()
    except Exception:
        pass
    return ""


def _extract_json_object(content: str) -> dict:
    """Parse content as a JSON object, or extract the first {...} block."""
    import re as _re
    content = (content or "").strip()
    try:
        obj = json.loads(content)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    m = _re.search(r"\{.*\}", content, flags=_re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return {}


_COMPLETE_SYSTEM = (
    "You are a text-to-video MOTION prompt writer for an AI film pipeline. "
    "The user gives a scene's FIRST-FRAME (keyframe) image prompt, its "
    "characters and any spoken dialogue. Write ONLY:\n"
    "1. video_prompt: a continuous MOTION prompt for the ~4 second clip — "
    "camera movement plus how the characters/elements move, STARTING from "
    "that exact first frame. It must be DIFFERENT from the image prompt — "
    "focus on ACTION, CAMERA and MOTION, not the static scene setup. 1-2 "
    "sentences, present tense.\n"
    "2. audio_lines: one short ambient-sound / narration line for the scene's "
    "audio (or an empty string if the scene is silent; if dialogue is present, "
    "audio_lines should be the ambient sound only, not the dialogue).\n"
    "Reply with STRICT JSON ONLY, no markdown, no extra words:\n"
    '{"video_prompt": "...", "audio_lines": "..."}'
)


def _complete_worker(story, limit):
    scenes = (story or {}).get("scenes") or []
    model, warning = _pick_llm_model()
    with _complete_lock:
        _complete.update(state="running", mode="fill", model=model,
                         warning=warning, errors=[], filled=0, done=0,
                         total=0, current="", started=time.time())
    # scenes that need a distinct motion prompt
    need_idx = []
    for sc in scenes or []:
        img = str(sc.get("image_prompt") or "").strip()
        vid = str(sc.get("video_prompt") or "").strip()
        same = bool(vid) and vid.strip().lower() == img.strip().lower()
        if not vid or same:
            need_idx.append(sc.get("index"))
    if limit:
        need_idx = need_idx[:int(limit)]
    with _complete_lock:
        _complete["total"] = len(need_idx)
    updated = {}
    done = 0
    filled = 0
    for idx in need_idx:
        if _cancel_requested():
            break
        sc = next((s for s in scenes if s.get("index") == idx), None)
        if sc is None:
            continue
        sc = dict(sc)
        with _complete_lock:
            _complete["current"] = f"Scene {idx}"
        try:
            img = str(sc.get("image_prompt") or "").strip()
            chars = ", ".join(str(c).strip()
                              for c in (sc.get("characters_present") or []))
            dia = str(sc.get("dialogue") or "").strip() or "(none)"
            content = _llm_chat([
                {"role": "system", "content": _COMPLETE_SYSTEM},
                {"role": "user", "content":
                    f"Scene {idx}\n"
                    f"Image prompt (first frame): {img}\n"
                    f"Characters present: {chars or '(none)'}\n"
                    f"Dialogue: {dia}"},
            ], model)
            vp = _extract_json_string(content, "video_prompt")
            audio = _extract_json_string(content, "audio_lines")
            if vp:
                sc["video_prompt"] = vp
                filled += 1
            if audio:
                sc["audio_lines"] = audio
            if not vp:
                with _complete_lock:
                    _complete["errors"].append(
                        f"Scene {idx}: LLM returned no usable video prompt")
        except Exception as exc:
            with _complete_lock:
                _complete["errors"].append(f"Scene {idx}: {exc}")
        updated[idx] = sc
        done += 1
        with _complete_lock:
            _complete["done"] = done
            _complete["filled"] = filled
    if _cancel_requested():
        with _complete_lock:
            _complete["state"] = "cancelled"
            _complete["current"] = ""
        return
    # merge back into the full scene list and persist (story.json + custom_story.txt)
    merged = []
    for sc in scenes or []:
        merged.append(updated.get(sc.get("index"), sc))
    out = {"story_title": (story or {}).get("story_title", "Custom story"),
           "scenes": merged}
    try:
        _write_story(out)
    except Exception as exc:
        with _complete_lock:
            _complete["errors"].append(f"save: {exc}")
    with _complete_lock:
        _complete["state"] = "done"
        _complete["current"] = ""


def start_story_completion(story=None, limit=None) -> tuple:
    """Start the LLM story-completion worker. Returns (error_msg, ok)."""
    with _complete_lock:
        if (_complete["thread"] is not None
                and _complete["thread"].is_alive()):
            return "a story completion is already running", False
        if story is None:
            story = _load_story()
        _complete["cancel"] = False
        _complete["thread"] = threading.Thread(
            target=_complete_worker, args=(story, limit), daemon=True)
        _complete["thread"].start()
        return "", True


# --------------------------------------------------- full story generation
_GEN_SYSTEM = (
    "You are a film storyboard writer for an AI text-to-image + text-to-video "
    "pipeline. The user gives you a GENRE / PREMISE, a scene count N, and a "
    "duration S per scene. Follow that premise EXACTLY — write the story ABOUT "
    "what the user describes, never a different subject.\n"
    "PLAN THE LOCATIONS YOURSELF: the story is shot in a small number of "
    "distinct locations (backgrounds) that YOU decide based on the story's "
    "needs. Like a real film, each location is shot over MULTIPLE consecutive "
    "scenes: establish the place once, then show different camera angles and "
    "actions in that same place before moving on. Do NOT invent a brand-new "
    "location for every scene — a short story (2-6 scenes) may use ~1-3 "
    "locations, a longer one (15-30 scenes) ~5-8. Decide how many scenes "
    "share each background and keep them locked.\n"
    "Structure the story in clear acts:\n"
    "  - setup (first ~20%): introduce the setting and the characters\n"
    "  - escalation (next ~20%): events build and new complications arise\n"
    "  - pursuit / confrontation (middle ~30%): the main character tries "
    "DIFFERENT strategies with setbacks\n"
    "  - climax (late ~15%): the final struggle\n"
    "  - resolution (last ~15%): things settle back to order\n"
    "CRITICAL RULES:\n"
    "  - GROUP SCENES BY LOCATION: consecutive scenes should often share ONE "
    "background. Only change to a new location occasionally, when the story "
    "demands it.\n"
    "  - LOCKED BACKGROUND: every scene in the same location MUST reuse the "
    "EXACT same location/background description in its image_prompt — the "
    "same room/place, same props, same decor, same lighting — and change ONLY "
    "the camera angle, the characters' position, and the action. Never "
    "reword or replace the background.\n"
    "  - Put the pure location description (the room/place, its fixed props, "
    "decor and lighting — NO characters, NO camera, NO action) in the "
    "'background' field, and repeat it verbatim inside every image_prompt of "
    "that location.\n"
    "  - Each scene must show a DIFFERENT action / story beat within its "
    "location. Never repeat the same beat twice; use fresh, inventive ways to "
    "advance the story.\n"
    "  - Vary who is present and vary camera ideas between scenes (wide, "
    "close-up, low-angle, crane, dolly, tracking, overhead, etc.).\n"
    "  - Keep the main characters' appearance consistent across every scene "
    "and keep locations logically connected.\n"
    "  - Match your pacing to N scenes × S seconds each (the user tells you "
    "N and S): a scene's script/action must fill roughly S seconds of video.\n"
    "For each scene you are asked for individually, reply with STRICT JSON "
    "only (the API forces JSON format):\n"
    '{"location": "short location name", "background": "pure fixed-location '
    'description (room/place, props, decor, lighting; NO characters or '
    'camera)", "image_prompt": "cinematic first-frame keyframe description '
    'that INCLUDES the exact background text verbatim, plus camera framing", '
    '"video_prompt": "continuous MOTION description for the clip: camera move '
    'plus how characters/elements move, DIFFERENT from image_prompt, ~S s", '
    '"characters_present": ["names of characters visible in this scene"], '
    '"dialogue": "Speaker: line" or null, "audio_lines": "short ambient '
    'sound cue"}\n'
    "For scene 1 only, you may also include \"story_title\": \"a short title "
    'for the film".'
)


_GEN_SCHEMA_HINT = (
    '{"location": "...", "background": "...", "image_prompt": "...", '
    '"video_prompt": "...", "characters_present": ["..."], '
    '"dialogue": "..." or null, "audio_lines": "..."}'
)


def _loc_key(name: str) -> str:
    """Normalize a location name for grouping (lowercase, collapse spaces,
    drop a leading article) so 'The Bedroom' and 'bedroom' map together."""
    import re as _re
    name = _re.sub(r"^(the|a|an)\s+", "", (name or "").strip().lower())
    return _re.sub(r"\s+", " ", name).strip()


def _generate_worker(params, limit):
    model, warning = _pick_llm_model()
    with _complete_lock:
        _complete.update(state="running", mode="generate", model=model,
                         warning=warning, errors=[], filled=0, done=0,
                         total=0, current="", started=time.time())
    genre = str((params or {}).get("genre", "") or "cinematic").strip()
    n = max(1, int((params or {}).get("num_scenes", 24)))
    seconds = float((params or {}).get("seconds", 4))
    if limit:
        n = min(n, int(limit))
    total_secs = n * seconds
    with _complete_lock:
        _complete["total"] = n
    scenes = []
    ok_count = 0
    # LLM-PLANNED LOCATIONS: the LLM decides how many backgrounds the story
    # needs and which scenes share each one. Every scene that reuses a location
    # is locked to that location's canonical background (verbatim).
    loc_backgrounds = {}   # _loc_key(name) -> canonical background text
    loc_display = {}       # _loc_key(name) -> canonical display name
    loc_scene_count = {}   # _loc_key(name) -> scenes so far in this location
    loc_order = []         # canonical display names, in introduction order
    used_summ = []
    story_title = "LLM story"
    for i in range(1, n + 1):
        if _cancel_requested():
            break
        with _complete_lock:
            _complete["current"] = f"Scene {i}"
        obj = None
        for attempt in range(1, 4):  # retry a flaky LLM call up to 3 times
            try:
                lines = [
                    f"Genre: {genre}",
                    f"Scene {i} of {n} — each scene is ~{seconds}s, so the "
                    f"whole film runs ~{total_secs}s. Write this scene's script "
                    f"to fill its {seconds}s.",
                ]
                if i == 1:
                    lines.append("This is the opening scene. Establish the "
                                 "story's FIRST location clearly.")
                else:
                    # Feed the LLM its own running location plan so it keeps
                    # several scenes in one place and only moves when the story
                    # truly changes location.
                    if loc_order:
                        lines.append("LOCATION PLAN SO FAR (reuse one of these "
                                     "for this scene when natural; introduce a "
                                     "NEW location only when the story really "
                                     "moves somewhere new):")
                        for name in loc_order:
                            k = _loc_key(name)
                            lines.append(f"  - {name} "
                                         f"({loc_scene_count.get(k, 0)} scenes) — "
                                         f"{loc_backgrounds.get(k, '')[:180]}")
                    else:
                        lines.append("No location established yet.")
                if used_summ:
                    lines.append("Previous scenes - this scene MUST show a "
                                 "DIFFERENT action/beat from ALL of these:")
                    lines.extend(f"  - scene {k}: {s}"
                                 for k, s in enumerate(used_summ, start=1))
                lines.append(f"Now write scene {i}. Return the strict JSON object "
                             f"for this scene only, shape {_GEN_SCHEMA_HINT}")
                user = "\n".join(lines)
                content = _llm_chat([
                    {"role": "system", "content": _GEN_SYSTEM},
                    {"role": "user", "content": user},
                ], model, json_format=True)
                obj = _extract_json_object(content)
                if obj and str(obj.get("image_prompt") or "").strip():
                    break
            except Exception as exc:
                with _complete_lock:
                    _complete["errors"].append(f"Scene {i} (try {attempt}): {exc}")
        img = str((obj or {}).get("image_prompt") or "").strip()
        if not img:
            # Keep the scene count exact: fall back to the most recent
            # location's locked background (or a generic prompt).
            with _complete_lock:
                _complete["errors"].append(
                    f"Scene {i}: no usable image_prompt after 3 tries — using fallback")
            bg = ""
            loc = ""
            if loc_order:
                lastk = _loc_key(loc_order[-1])
                bg = loc_backgrounds.get(lastk, "")
                loc = loc_order[-1]
            scenes.append({
                "index": i,
                "location": loc or f"Location {len(loc_order) + 1}",
                "background": bg,
                "image_prompt": ((f"{bg}. " if bg else "")
                                 + f"{genre} — scene {i}, cinematic detailed first "
                                 f"frame, atmospheric lighting, moody."),
                "video_prompt": (f"A gentle dolly move through the scene as it "
                                 f"plays out, {genre}, about {seconds}s."),
                "characters_present": [],
                "dialogue": None,
                "audio_lines": "",
            })
            used_summ.append(f"scene {i} | fallback scene")
        else:
            if i == 1:
                t = str((obj or {}).get("story_title") or "").strip()
                if t:
                    story_title = t
            loc = str((obj or {}).get("location") or "").strip()
            bg = str((obj or {}).get("background") or "").strip()
            k = _loc_key(loc)
            if not k:
                # LLM gave no location name: reuse the most recent one.
                if loc_order:
                    k = _loc_key(loc_order[-1])
                    loc = loc_order[-1]
                else:
                    loc = "Location 1"
                    k = _loc_key(loc)
            if k in loc_backgrounds:
                # REUSE an established location -> lock its canonical bg.
                canonical = loc_backgrounds[k]
                if canonical and canonical not in img:
                    img = f"{canonical}. {img}"
                bg = canonical
                loc = loc_display[k]   # keep the same name for the location
                loc_scene_count[k] = loc_scene_count.get(k, 0) + 1
            else:
                # NEW location -> establish its canonical bg from this scene.
                if not bg:
                    bg = img
                loc_backgrounds[k] = bg
                loc_display[k] = loc
                loc_scene_count[k] = 1
                loc_order.append(loc)
            chars = (obj or {}).get("characters_present") or []
            if isinstance(chars, str):
                chars = [chars]
            chars = [str(c).strip() for c in chars if str(c).strip()]
            vid = str((obj or {}).get("video_prompt") or img).strip()
            dia = str((obj or {}).get("dialogue") or "").strip() or None
            audio = str((obj or {}).get("audio_lines") or "").strip()
            scenes.append({
                "index": i,
                "location": loc,
                "background": loc_backgrounds.get(k, bg),
                "image_prompt": img,
                "video_prompt": vid,
                "characters_present": chars,
                "dialogue": dia,
                "audio_lines": audio,
            })
            used_summ.append((loc or f"location {i}") + " | " + img[:200])
            ok_count += 1
            with _complete_lock:
                _complete["filled"] = ok_count
        with _complete_lock:
            _complete["done"] = i
    # Guarantee background lock by location NAME: every scene carries its
    # location's canonical background verbatim in its image_prompt.
    for sc in scenes:
        k = _loc_key(str(sc.get("location") or ""))
        bg = loc_backgrounds.get(k, "") if k else ""
        if bg:
            if sc.get("background") != bg:
                sc["background"] = bg
            img = str(sc.get("image_prompt") or "")
            if bg not in img:
                sc["image_prompt"] = f"{bg}. {img}"
            if not sc.get("location"):
                sc["location"] = loc_display.get(k, "")
    if _cancel_requested():
        with _complete_lock:
            _complete["state"] = "cancelled"
            _complete["current"] = ""
        return
    try:
        _write_story({"story_title": story_title, "scenes": scenes})
    except Exception as exc:
        with _complete_lock:
            _complete["errors"].append(f"save: {exc}")
    with _complete_lock:
        _complete["state"] = "done"
        _complete["current"] = ""


def _custom_freeform_premise() -> str:
    """If custom_story.txt holds free-form prose (no IMAGE:/VIDEO: scene
    blocks), return it as the generation PREMISE so the user's own words drive
    the story. Scene-block text is a finished story (direct bypass), not a
    premise, so return "". Comment lines are ignored."""
    import re as _re
    cfg = _read_config().get("story", {})
    fname = str(cfg.get("custom_story_file", "custom_story.txt")).replace("..", "")
    path = os.path.join(BASE_DIR, fname)
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()
    except Exception:
        return ""
    content = [ln.strip() for ln in lines
               if ln.strip() and not ln.strip().startswith("#")]
    if not content:
        return ""
    text = "\n".join(content).strip()
    if _re.search(r"^\s*(image|video|characters|dialogue|voice):",
                  text, _re.M | _re.I):
        return ""  # finished scene-block story, not a premise
    return text if len(text) >= 20 else ""


def start_story_generation(params=None, limit=None) -> tuple:
    """Start the LLM story-generation worker. Returns (error_msg, ok)."""
    with _complete_lock:
        if (_complete["thread"] is not None
                and _complete["thread"].is_alive()):
            return "a story task is already running", False
        story_cfg = _read_config().get("story", {})
        if params is None:
            params = {}
        # Prefer the caller's explicit values; fill any missing ones from config.
        params = {
            "genre": str(params.get("genre") or story_cfg.get("genre", "")
                         or "").strip(),
            "num_scenes": max(1, int(params.get("num_scenes")
                                     or story_cfg.get("scenes", 24))),
            "seconds": max(1.0, float(params.get("seconds")
                                      or story_cfg.get("seconds_per_scene", 4))),
        }
        # If the user left a free-form description in the Custom Story editor,
        # THEIR words are the premise — it wins over the stale genre field.
        premise = _custom_freeform_premise()
        if premise:
            params["genre"] = premise
        _complete["cancel"] = False
        _complete["thread"] = threading.Thread(
            target=_generate_worker, args=(params, limit), daemon=True)
        _complete["thread"].start()
        return "", True


def _complete_status() -> dict:
    with _complete_lock:
        return {
            "state": _complete["state"],
            "mode": _complete["mode"],
            "total": _complete["total"],
            "done": _complete["done"],
            "filled": _complete["filled"],
            "errors": list(_complete["errors"]),
            "model": _complete["model"],
            "warning": _complete["warning"],
            "current": _complete["current"],
            "started": _complete["started"],
            "cancel": _complete["cancel"],
        }


def _load_video_score():
    """Import director/video_score.py robustly regardless of cwd."""
    import importlib.util
    path = os.path.join(BASE_DIR, "video_score.py")
    spec = importlib.util.spec_from_file_location("director_video_score", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create module spec for {path}")
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


def _director_running(force: bool = False) -> bool:
    """True if a director.py process is active anywhere (terminal or dashboard).
    Cached ~5s to avoid hammering the OS on every health poll (unless force)."""
    now = time.time()
    if not force and now - _scan_cache["t"] < 5:
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
    for s in streams:
        for line in iter(s.readline, ""):
            text = line.rstrip("\n")
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
            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
            bufsize=1, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except Exception as exc:
        with _run_lock:
            _run["stdout"].append(f"[controller] failed to start: {exc}")
            _run["launching"] = False
        return
    with _run_lock:
        _run["proc"] = proc
        _run["launching"] = False
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


def start_run(scene=None, scene_count=None) -> str:
    with _run_lock:
        if _run.get("launching"):
            return "A pipeline run is already starting — please wait a moment."
        if _run["proc"] is not None and _run["proc"].poll() is None:
            return "A pipeline run is already active — wait for it to finish or press Stop first."
        # Reserve the slot ATOMICALLY before spawning the thread. Without this,
        # two rapid /api/run calls can BOTH pass the check above and launch two
        # concurrent director.py processes, which clobber each other's output
        # and hang at startup (blocking every later run with a 409). The flag is
        # cleared in _runner once the process is up (or the spawn fails).
        _run["launching"] = True
    # Fresh OS scan (bypass the 5s health cache): refuse to start if ANY
    # director.py is already active, even one launched from a terminal.
    if _director_running(force=True):
        with _run_lock:
            _run["launching"] = False
        return "A pipeline run is already active on this machine — wait for it to finish or press Stop first."
    stop_flag = threading.Event()
    args = ["--config", "config.json"]
    if scene is not None:
        args += ["--scene", str(scene)]
    if scene_count is not None:
        args += ["--scene-count", str(scene_count)]
    t = threading.Thread(target=_runner, args=(args, stop_flag), daemon=True)
    with _run_lock:
        _run["thread"] = t
        _run["_stop_flag"] = stop_flag
    try:
        t.start()
    except Exception:
        with _run_lock:
            _run["launching"] = False
        return "Failed to start run thread."
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
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionAbortedError):
            pass  # client closed/reloaded the page mid-response; not an error

    def _file(self, path, content_type):
        try:
            with open(path, "rb") as f:
                body = f.read()
        except Exception:
            self.send_error(404)
            return
        try:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionAbortedError):
            pass  # client closed/reloaded the page mid-response; not an error

    def log_message(self, format: str, *args) -> None:  # quiet
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
        elif path == "/api/story":
            self._json({"story": _load_story()})
        elif path == "/api/custom_story":
            cfg = _read_config().get("story", {})
            fname = str(cfg.get("custom_story_file",
                                "custom_story.txt")).replace("..", "")
            fpath = os.path.join(BASE_DIR, fname)
            try:
                with open(fpath, encoding="utf-8") as f:
                    text = f.read()
            except Exception:
                text = ""
            self._json({"file": fname, "text": text})
        elif path == "/api/story/complete_status":
            self._json(_complete_status())
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
        elif path == "/api/story":
            story = data.get("story")
            if not isinstance(story, dict):
                self._json({"error": "story object required"}, 400)
                return
            try:
                _write_story(story)
                self._json({"ok": True})
            except Exception as exc:
                self._json({"error": str(exc)}, 500)
        elif path == "/api/story/complete":
            story = data.get("story")
            if not isinstance(story, dict):
                self._json({"error": "story object required"}, 400)
                return
            limit = data.get("limit")
            msg, ok = start_story_completion(story=story, limit=limit)
            self._json({"ok": ok, "message": msg or "started"},
                       200 if ok else 409)
        elif path == "/api/story/generate":
            params = data.get("params")
            limit = data.get("limit")
            msg, ok = start_story_generation(params=params, limit=limit)
            self._json({"ok": ok, "message": msg or "started"},
                       200 if ok else 409)
        elif path == "/api/story/cancel":
            with _complete_lock:
                _complete["cancel"] = True
                running = _complete["state"] == "running"
            self._json({"ok": running,
                        "message": "cancelling" if running else "no task running"})
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
            scene_count = data.get("scene_count")
            scene_count = int(scene_count) if scene_count is not None else None
            msg = start_run(scene=scene, scene_count=scene_count)
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
