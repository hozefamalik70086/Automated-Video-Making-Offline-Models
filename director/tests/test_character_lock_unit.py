#!/usr/bin/env python3
"""Unit tests for the character-locked pipeline.

Covers the NEW feature only:
  * storywriter custom-story parsing (CHARACTERS:/DIALOGUE:/VOICE:)
  * fallback story schema (characters_present / dialogue)
  * characters.py offline logic (extract, locks, voices, ref workflow,
    scene blocks, consistency report)
  * director._scene_values character injection
  * narrate.make_narration per-character voice selection

Run (GPU-free, no network, llm.backend == "none"):
    cd director
    python tests/test_character_lock_unit.py

Exit code 0 = all passed.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tempfile
import traceback

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)
os.chdir(BASE)

from storywriter import StoryWriter, FALLBACK_STORY  # noqa: E402
import characters as chars  # noqa: E402

PASS = 0
FAIL = 0
FAILURES = []


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        FAILURES.append(name)
        print(f"  FAIL  {name}  {detail}")


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def base_config() -> dict:
    """Minimal config with llm off and a temp character library dir."""
    cfg = json.loads(json.dumps({
        "comfyui": {"url": "http://127.0.0.1:8899", "timeout_seconds": 5},
        "llm": {"backend": "none"},
        "story": {"custom_story_file": "tests/nonexistent.txt",
                  "genre": "test", "scenes": 4, "seconds_per_scene": 5},
        "render": {"video_width": 480, "video_height": 280, "fps": 25,
                   "image_width": 1080, "image_height": 720,
                   "image_steps": 20, "camera_move": "auto"},
        "qc": {"enabled": True, "max_attempts": 3},
        "director": {"output_dir": "tests/output_test",
                     "workflow_template": "workflow_scene_template.json",
                     "workflow_knobs": "workflow_knobs.json"},
        "audio": {"enabled": True, "engine": "edge-tts",
                  "voice": "en-US-GuyNeural", "rate": "+0%"},
        "characters": {"enabled": True, "auto_generate_refs": True,
                       "library_dir": "characters",
                       "ref_template": "workflow_scene_template.json",
                       "consistency_threshold": 0.85},
    }))
    return cfg


# --------------------------------------------------------------------------- #
#  storywriter custom story parsing
# --------------------------------------------------------------------------- #
def test_custom_story_parsing() -> None:
    section("storywriter: custom story parsing (real custom_story.txt)")
    cfg = base_config()
    cfg["story"]["custom_story_file"] = "custom_story.txt"
    cfg["story"]["scenes"] = 24
    sw = StoryWriter(cfg, BASE)
    text = sw._read_custom()
    check("custom_story.txt is used (non-empty)", bool(text))
    parsed = sw._parse_custom(text)
    scenes = parsed["scenes"]
    check("24 scenes parsed", len(scenes) == 24, f"got {len(scenes)}")
    # No 'Shot List' or 'Clip N' pollution in any prompt
    pollution = [s for s in scenes
                 if re.search(r"\b(shot list|clip\s+\d)\b",
                              (s["image_prompt"] + " " + s["video_prompt"]).lower())]
    check("no Shot List / Clip N pollution", not pollution,
          f"found {len(pollution)}")
    # Scene 4 characters
    check("scene 4 characters present",
          scenes[3]["characters_present"] == ["Red Umbrella", "Toddler"],
          repr(scenes[3]["characters_present"]))
    # Scene 22 dialogue
    check("scene 22 has dialogue",
          (scenes[21]["dialogue"] or "").startswith("Little Girl:"),
          repr(scenes[21]["dialogue"]))
    check("scene 22 dialogue speaker added to characters",
          "Little Girl" in scenes[21]["characters_present"],
          repr(scenes[21]["characters_present"]))
    # Every scene declares characters
    empty = [i for i, s in enumerate(scenes, 1) if not s["characters_present"]]
    check("every scene declares CHARACTERS", not empty, f"empty: {empty}")
    # finalize preserves
    fin = sw._finalize(parsed)
    check("finalize keeps 24 scenes (config scenes=24)",
          len(fin["scenes"]) == 24, f"got {len(fin['scenes'])}")
    check("finalize preserves characters_present",
          fin["scenes"][3]["characters_present"] == ["Red Umbrella", "Toddler"])
    check("finalize preserves dialogue",
          (fin["scenes"][21]["dialogue"] or "").startswith("Little Girl:"))


def test_custom_inline_dialogue() -> None:
    section("storywriter: inline 'Name: line' dialogue (declared chars)")
    cfg = base_config()
    sw = StoryWriter(cfg, BASE)
    text = (
        "CHARACTERS: Hero, Dog\n"
        "Hero: We made it!\n"
        "The hero and his dog reach the summit.\n"
    )
    parsed = sw._parse_custom(text)
    sc = parsed["scenes"][0]
    check("inline dialogue captured",
          (sc["dialogue"] or "").startswith("Hero:"), repr(sc["dialogue"]))
    check("speaker added to characters",
          "Hero" in sc["characters_present"])
    check("body not polluted by dialogue line",
          "We made it" not in sc["image_prompt"], repr(sc["image_prompt"]))

    # Without CHARACTERS, a "Name:" line must NOT be treated as dialogue
    text2 = "The hero climbs the hill.\n"
    parsed2 = sw._parse_custom(text2)
    check("no dialogue when no characters declared",
          parsed2["scenes"][0]["dialogue"] is None)


def test_fallback_story_schema() -> None:
    section("storywriter: fallback story schema")
    check("FALLBACK_STORY is a dict", isinstance(FALLBACK_STORY, dict))
    scenes = FALLBACK_STORY.get("scenes", [])
    check("fallback has scenes", len(scenes) >= 1)
    s0 = scenes[0]
    check("fallback scene has characters_present",
          isinstance(s0.get("characters_present"), list))
    check("fallback scene has dialogue key", "dialogue" in s0)
    check("fallback scene has audio_lines key", "audio_lines" in s0)


# --------------------------------------------------------------------------- #
#  characters.py offline logic
# --------------------------------------------------------------------------- #
def test_helpers() -> None:
    section("characters: helper functions")
    check("_norm lowercases + collapses", chars._norm("  Little  Girl  ") == "little girl")
    check("_dialogue_speaker basic", chars._dialogue_speaker("Little Girl: hi") == "Little Girl")
    check("_dialogue_speaker pipe", chars._dialogue_speaker("Hero| hey") == "Hero")
    check("_dialogue_speaker guards camera",
          chars._dialogue_speaker("Camera pans left") is None)
    check("_dialogue_speaker guards narrator",
          chars._dialogue_speaker("Narrator: once upon a time") is None)
    check("_dialogue_speaker guards close-up",
          chars._dialogue_speaker("Close-up of the umbrella") is None)
    check("_dialogue_speaker empty", chars._dialogue_speaker("") is None)
    # voice heuristic
    check("voice: female child -> Ana",
          chars._pick_edge_voice("a female child", "") == "en-US-AnaNeural")
    check("voice: little girl -> Ana",
          chars._pick_edge_voice("", "Little Girl") == "en-US-AnaNeural")
    check("voice: elderly grandmother -> Michelle",
          chars._pick_edge_voice("elderly grandmother", "") == "en-US-MichelleNeural")
    check("voice: adult male -> Brian",
          chars._pick_edge_voice("deep adult male", "") == "en-US-BrianNeural")
    check("voice: young woman -> Aria",
          chars._pick_edge_voice("young woman", "") == "en-US-AriaNeural")
    check("voice: default (genderless) -> adult male Brian",
          chars._pick_edge_voice("robot", "") == "en-US-BrianNeural")


def test_extract_characters() -> None:
    section("characters: extract_characters (Phase 0)")
    story = {
        "story_title": "Test",
        "scenes": [
            {"characters_present": ["Red Umbrella"], "dialogue": None},
            {"characters_present": ["Red Umbrella", "Little Girl"],
             "dialogue": "Little Girl: Grandma, look what I found!"},
        ],
    }
    cast = chars.CharacterLibrary(base_config(), BASE).extract_characters(story)
    names = {c["name"]: c for c in cast}
    check("cast has Red Umbrella", "Red Umbrella" in names)
    check("cast has Little Girl", "Little Girl" in names)
    check("Little Girl marked speaking",
          names["Little Girl"]["speaking"] is True)
    check("Red Umbrella not speaking",
          names["Red Umbrella"]["speaking"] is False)
    check("cast ids sequential",
          [c["id"] for c in cast] == ["char_001", "char_002"])


def test_ensure_locks_offline() -> None:
    section("characters: ensure_locks offline (llm none, no refs)")
    tmp = tempfile.mkdtemp(prefix="char_test_")
    try:
        cfg = base_config()
        cfg["characters"]["library_dir"] = os.path.join(
            "tests", "._char_test_lib")
        story = {
            "story_title": "Test",
            "scenes": [
                {"characters_present": ["Red Umbrella"], "dialogue": None},
                {"characters_present": ["Red Umbrella", "Little Girl"],
                 "dialogue": "Little Girl: Grandma, look what I found!"},
            ],
        }
        lib = chars.CharacterLibrary(cfg, BASE)
        lib.lib_dir = tmp
        lib.refs_dir = os.path.join(tmp, "refs")
        os.makedirs(lib.refs_dir, exist_ok=True)
        lib.library_path = os.path.join(tmp, "characters.json")
        locks = lib.ensure_locks(story, generate_refs=False)
        check("2 locks created", len(locks) == 2, f"got {len(locks)}")
        by_name = {l["name"]: l for l in locks}
        check("fallback visual descriptor written",
              by_name["Red Umbrella"]["visual_descriptor"].startswith(
                  "A clearly visible Red Umbrella"))
        check("Little Girl has voice_lock",
              by_name["Little Girl"]["voice_lock"] is not None)
        check("Little Girl voice is child female",
              by_name["Little Girl"]["voice_lock"]["voice"] == "en-US-AnaNeural",
              repr(by_name["Little Girl"]["voice_lock"]))
        check("Red Umbrella has no voice_lock",
              by_name["Red Umbrella"]["voice_lock"] is None)
        check("library file saved", os.path.exists(lib.library_path))
        # reuse by name -> no new lock on second run
        locks2 = lib.ensure_locks(story, generate_refs=False)
        check("reuse: same 2 locks (no duplicates)",
              len(locks2) == 2 and len(lib.library) == 2,
              f"library len={len(lib.library)}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_build_ref_workflow() -> None:
    section("characters: build_ref_workflow (T2I subgraph)")
    lib = chars.CharacterLibrary(base_config(), BASE)
    wf = lib.build_ref_workflow("A red umbrella", 12345, "director/charref/char_001")
    check("T2I nodes present", all(n in wf for n in chars.REF_NODE_IDS))
    check("SaveImage node 60 present", chars.REF_SAVE_NODE in wf)
    check("node 6 text set", wf["6"]["inputs"]["text"] == "A red umbrella")
    check("node 9 seed set", wf["9"]["inputs"]["seed"] == 12345)
    check("node 60 wired to node 11",
          wf[chars.REF_SAVE_NODE]["inputs"]["images"] == ["11", 0])
    check("node 60 has filename_prefix",
          wf[chars.REF_SAVE_NODE]["inputs"]["filename_prefix"]
          == "director/charref/char_001")


def test_scene_blocks_and_voice() -> None:
    section("characters: scene_character_blocks + voice_for_scene (Phase 3)")
    tmp = tempfile.mkdtemp(prefix="char_test_")
    try:
        cfg = base_config()
        story = {
            "story_title": "Test",
            "scenes": [
                {"characters_present": ["Red Umbrella"], "dialogue": None},
                {"characters_present": ["Red Umbrella", "Little Girl"],
                 "dialogue": "Little Girl: Grandma, look what I found!"},
            ],
        }
        lib = chars.CharacterLibrary(cfg, BASE)
        lib.lib_dir = tmp
        lib.refs_dir = os.path.join(tmp, "refs")
        os.makedirs(lib.refs_dir, exist_ok=True)
        lib.library_path = os.path.join(tmp, "characters.json")
        lib.ensure_locks(story, generate_refs=False)
        img, note = lib.scene_character_blocks(story["scenes"][0])
        check("image block for Red Umbrella", len(img) == 1)
        check("video note mentions characters",
              "Red Umbrella" in note and "Characters present" in note)
        # scene 2: dialogue speaker gets the locked voice
        vl = lib.voice_for_scene(story["scenes"][1])
        check("voice_for_scene returns Little Girl's voice",
              vl is not None and vl["voice"] == "en-US-AnaNeural", repr(vl))
        # scene 1: no dialogue -> falls to first speaking char in scene -> none
        vl1 = lib.voice_for_scene(story["scenes"][0])
        check("voice_for_scene scene 1 -> None", vl1 is None)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_consistency_report() -> None:
    section("characters: write_consistency_report (Phase 4)")
    tmp = tempfile.mkdtemp(prefix="char_test_")
    try:
        cfg = base_config()
        story = {
            "story_title": "Test",
            "scenes": [
                {"characters_present": ["Red Umbrella"], "dialogue": None},
                {"characters_present": ["Red Umbrella", "Little Girl"],
                 "dialogue": "Little Girl: hi"},
            ],
        }
        lib = chars.CharacterLibrary(cfg, BASE)
        lib.lib_dir = tmp
        lib.refs_dir = os.path.join(tmp, "refs")
        os.makedirs(lib.refs_dir, exist_ok=True)
        lib.library_path = os.path.join(tmp, "characters.json")
        lib.ensure_locks(story, generate_refs=False)
        red = next(l for l in lib.library if l["name"] == "Red Umbrella")
        full_desc = red["visual_descriptor"]
        results = [
            {"scene": 1, "image_prompt": story["scenes"][0]["characters_present"][0]
             + " " + full_desc},
            {"scene": 2, "image_prompt": "x y z"},
        ]
        report = lib.write_consistency_report(story, results, tmp)
        out = os.path.join(tmp, "consistency_report.json")
        check("report file written", os.path.exists(out))
        check("report has characters", len(report["characters"]) == 2)
        check("face check status reported",
              "face_embedding_check" in report)
        ru = next(c for c in report["characters"]
                  if c["name"] == "Red Umbrella")
        check("Red Umbrella scenes_present == [1,2]",
              ru["scenes_present"] == [1, 2], repr(ru["scenes_present"]))
        anchors = {a["scene"]: a["anchor_in_prompt"] for a in ru["prompt_anchor_check"]}
        check("anchor detected in scene 1 prompt", anchors.get(1) is True)
        check("anchor absent in scene 2 prompt", anchors.get(2) is False)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- #
#  director._scene_values injection
# --------------------------------------------------------------------------- #
def test_director_scene_values() -> None:
    section("director: _scene_values character injection")
    import director as director_mod
    cfg = base_config()
    d = director_mod.Director(cfg)
    # inject a fake character library (no ComfyUI/LLM)
    class FakeLib:
        enabled = True

        def scene_character_blocks(self, scene):
            locks = {
                "Little Girl": "A 7-year-old girl with a gap-toothed smile, "
                               "bright yellow raincoat, red wellies.",
                "Red Umbrella": "A bright red umbrella with a black curved "
                                "wooden handle.",
            }
            blocks = [locks[n] for n in scene.get("characters_present", [])
                      if n in locks]
            names = [n for n in scene.get("characters_present", [])]
            note = ("Characters present (identical appearance to the "
                    "keyframe image): " + ", ".join(names) + ".")
            return blocks, note
    d.chars = FakeLib()
    scene = {
        "image_prompt": "A rainy street.",
        "video_prompt": "The camera pushes in as rain falls.",
        "characters_present": ["Little Girl", "Red Umbrella"],
    }
    vals = d._scene_values(scene, 1)
    ip = vals["image_prompt"]
    vp = vals["video_prompt"]
    check("descriptor injected into image prompt",
          "gap-toothed smile" in ip and "raincoat" in ip)
    check("consistency anchor injected", chars.CONSISTENCY_ANCHOR in ip)
    check("video note injected", "Characters present" in vp)
    check("duration uses config seconds", vals["video_duration"] == 5.0)
    check("save prefix per scene",
          vals["save_prefix"] == "director/scene_01")


# --------------------------------------------------------------------------- #
#  narrate.make_narration voice selection
# --------------------------------------------------------------------------- #
def test_narrate_voice_selection() -> None:
    section("narrate: per-character voice selection (logic only, no edge-tts)")
    import narrate as narrate_mod

    captured = []

    class FakeVoiceLock:
        def voice_for_scene(self, scene):
            return {"engine": "edge-tts", "voice": "en-US-AnaNeural",
                    "rate": "+0%"}

    class FakeLib:
        enabled = True
        voice_for_scene = FakeVoiceLock().voice_for_scene

    # monkeypatch edge_tts import to avoid network; capture calls
    class FakeEdge:
        class Communicate:
            def __init__(self, text, voice="", rate=""):
                captured.append({"text": text, "voice": voice, "rate": rate})

            async def save(self, path):
                with open(path, "w", encoding="utf-8") as f:
                    f.write("x")

    narrate_mod.edge_tts = FakeEdge
    # make_narration does `import edge_tts` and `from characters import
    # CharacterLibrary` INSIDE the function, so patch sys.modules / the
    # characters module attribute for the fake to take effect:
    import sys as _sys
    _orig_edge = _sys.modules.get("edge_tts")
    _sys.modules["edge_tts"] = FakeEdge
    orig_cl = chars.CharacterLibrary
    chars.CharacterLibrary = lambda cfg, base: FakeLib()
    tmp = tempfile.mkdtemp(prefix="narr_test_")
    try:
        report = {
            "scenes": [
                {"audio_lines": "", "dialogue": "Little Girl: Grandma, look!"},
                {"audio_lines": "gentle rain", "dialogue": None},
                {"audio_lines": "", "dialogue": None},
            ]
        }
        cfg = base_config()
        out = narrate_mod.make_narration(report, cfg, tmp)
        check("scene 1 spoken text is dialogue w/o prefix",
              bool(captured) and captured[0]["text"] == "Grandma, look!",
              repr(captured))
        check("scene 1 uses character voice",
              bool(captured) and captured[0]["voice"] == "en-US-AnaNeural",
              repr(captured))
        check("mp3 for scene 1",
              os.path.exists(os.path.join(tmp, "_narration", "scene_01.mp3")))
        check("scene 1 result is a path", bool(out[0]))
        check("scene 2 (audio line) also synthesized",
              bool(out[1]) and "scene_02" in out[1], repr(out[1]))
        check("scene 3 result None (no text)", out[2] is None)
    finally:
        if _orig_edge is None:
            _sys.modules.pop("edge_tts", None)
        else:
            _sys.modules["edge_tts"] = _orig_edge
        chars.CharacterLibrary = orig_cl
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- #
def main() -> None:
    print("Character-lock unit tests (offline, no GPU, no network)")
    test_custom_story_parsing()
    test_custom_inline_dialogue()
    test_fallback_story_schema()
    test_helpers()
    test_extract_characters()
    test_ensure_locks_offline()
    test_build_ref_workflow()
    test_scene_blocks_and_voice()
    test_consistency_report()
    test_director_scene_values()
    test_narrate_voice_selection()
    print(f"\n{'='*60}\nPASS: {PASS}   FAIL: {FAIL}")
    if FAILURES:
        print("Failures:")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(2)
