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
from director import Director  # noqa: E402
from qc import QCResult  # noqa: E402

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
    section("storywriter: custom story parsing")
    cfg = base_config()
    cfg["story"]["custom_story_file"] = "custom_story.txt"
    cfg["story"]["scenes"] = 24
    sw = StoryWriter(cfg, BASE)
    # Deterministic fixture (3 scene blocks) — independent of live file state,
    # because custom_story.txt may hold a free-form premise rather than a
    # finished scene-block story.
    fixture = (
        "# test fixture\n"
        "IMAGE: a cozy sunlit bedroom at noon, warm pastel light\n"
        "VIDEO: slow dolly-in past drifting dust motes\n"
        "CHARACTERS: Mia, Kofi\n"
        "DIALOGUE: Mia: finally, a lazy afternoon\n"
        "VOICE: gentle wind, birdsong\n"
        "---\n"
        "IMAGE: a rooftop garden with hanging plants swaying\n"
        "VIDEO: overhead crane shot tilting down to the skyline\n"
        "CHARACTERS: Kofi\n"
        "---\n"
        "A quiet kitchen in the evening. Mia stirs cocoa by lamplight.\n"
        "CHARACTERS: Mia, Kofi\n"
        "Kofi: the stars are out\n"
    )
    parsed = sw._parse_custom(fixture)
    scenes = parsed["scenes"]
    check("3 scenes parsed from fixture", len(scenes) == 3,
          f"got {len(scenes)}")
    # No 'Shot List' or 'Clip N' pollution in any prompt
    pollution = [s for s in scenes
                 if re.search(r"\b(shot list|clip\s+\d)\b",
                              (s["image_prompt"] + " " + s["video_prompt"]).lower())]
    check("no Shot List / Clip N pollution", not pollution,
          f"found {len(pollution)}")
    # A scene that declares characters lists them
    check("scene 2 declares characters",
          bool(scenes[1]["characters_present"]),
          repr(scenes[1]["characters_present"]))
    # At least one scene carries dialogue
    dia_idx = [i for i, s in enumerate(scenes, 1)
               if (s.get("dialogue") or "").strip()]
    check("at least one scene has dialogue", bool(dia_idx),
          f"dialogue scene indexes: {dia_idx}")
    # Dialogue speaker is added to that scene's characters
    if dia_idx:
        first = dia_idx[0] - 1
        m = re.match(r"\s*([^:]+):", scenes[first]["dialogue"])
        if m:
            name = m.group(1).strip()
            check("dialogue speaker added to characters",
                  name in scenes[first]["characters_present"],
                  f"{name!r} not in {scenes[first]['characters_present']!r}")
        else:
            check("dialogue speaker added to characters", False,
                  f"cannot parse speaker from {scenes[first]['dialogue']!r}")
    # Free-form body fills both image_prompt and video_prompt
    check("free-form body fills image & video prompt",
          scenes[2]["image_prompt"] == scenes[2]["video_prompt"],
          repr((scenes[2]["image_prompt"], scenes[2]["video_prompt"])))
    # Live custom_story.txt is still readable (state agnostic)
    live = sw._read_custom()
    check("custom_story.txt is read (non-empty)", bool(live))
    # Every scene declares characters
    empty = [i for i, s in enumerate(scenes, 1) if not s["characters_present"]]
    check("every scene declares CHARACTERS", not empty, f"empty: {empty}")
    # finalize preserves
    fin = sw._finalize(parsed)
    check("finalize keeps 24 scenes (config scenes=24)",
          len(fin["scenes"]) == 24, f"got {len(fin['scenes'])}")
    check("finalize preserves characters_present",
          fin["scenes"][2]["characters_present"]
          == scenes[2]["characters_present"],
          repr(fin["scenes"][2]["characters_present"]))
    if dia_idx:
        first = dia_idx[0] - 1
        check("finalize preserves dialogue",
              (fin["scenes"][first].get("dialogue") or "")
              == (scenes[first].get("dialogue") or ""),
              repr(fin["scenes"][first].get("dialogue")))
    else:
        check("finalize preserves dialogue", True, "no dialogue scenes")


def test_style_phrase_applies_to_both_prompts() -> None:
    section("director: style phrase propagation")
    cfg = base_config()
    cfg["story"]["genre"] = "Studio Ghibli (hand-drawn, soft watercolor)"
    cfg["story"]["character"] = ""
    cfg["characters"]["enabled"] = False
    director = Director(cfg)
    scene = {
        "image_prompt": "A quiet courtyard with a child under a tree",
        "video_prompt": "The camera slowly pans left across the courtyard",
    }
    values = director._scene_values(scene, 1)
    check("style phrase appended to image prompt",
          "studio ghibli" in values["image_prompt"].lower(),
          values["image_prompt"])
    check("style phrase appended to video prompt",
          "studio ghibli" in values["video_prompt"].lower(),
          values["video_prompt"])


def test_scene_count_selection() -> None:
    section("director: scene-count selection")
    cfg = base_config()
    director = Director(cfg)
    scenes = [{"image_prompt": "s1"}, {"image_prompt": "s2"}, {"image_prompt": "s3"}]
    selected, base_idx = director._select_scenes(scenes, scene_count=2)
    check("scene count returns first N scenes",
          [s["image_prompt"] for s in selected] == ["s1", "s2"],
          repr([s["image_prompt"] for s in selected]))
    check("scene count keeps base index at zero",
          base_idx == 0, str(base_idx))

    selected, base_idx = director._select_scenes(scenes, only_scene=1, scene_count=2)
    check("scene range starts at the chosen scene",
          [s["image_prompt"] for s in selected] == ["s2", "s3"],
          repr([s["image_prompt"] for s in selected]))
    check("scene range preserves the start index",
          base_idx == 1, str(base_idx))


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


def test_custom_story_multiline_dialogue() -> None:
    section("storywriter: multi-line dialogue preserved (not truncated)")
    cfg = base_config()
    sw = StoryWriter(cfg, BASE)
    text = (
        "IMAGE: banquet hall at night, warm candlelight\n"
        "VIDEO: slow dolly-in\n"
        "CHARACTERS: ELIZA, ANNA\n"
        "DIALOGUE: ELIZA: (Smiling sweetly) Oh! You know what they say.\n"
        "ANNA: Eliza, that's uncalled for.\n"
        "ELIZA: Uncalled for? It's just comedy!\n"
        "---\n"
    )
    parsed = sw._parse_custom(text)
    sc = parsed["scenes"][0]
    dlg = sc["dialogue"] or ""
    check("all 3 dialogue lines kept", dlg.count("\n") == 2, repr(dlg))
    check("first line kept",
          dlg.startswith("ELIZA: (Smiling sweetly)"), repr(dlg))
    check("middle line kept",
          "ANNA: Eliza, that's uncalled for." in dlg, repr(dlg))
    check("last line kept",
          "Uncalled for? It's just comedy!" in dlg, repr(dlg))
    check("all speakers added to characters",
          "ELIZA" in sc["characters_present"]
          and "ANNA" in sc["characters_present"],
          repr(sc["characters_present"]))


def test_normalize_preserves_dialogue() -> None:
    section("storywriter: _normalize keeps dialogue + characters_present")
    cfg = base_config()
    sw = StoryWriter(cfg, BASE)
    data = {"story_title": "Wedding", "scenes": [{
        "id": 1,
        "image_prompt": "img",
        "video_prompt": "vid",
        "characters_present": ["ELIZA", "ANNA"],
        "dialogue": "ELIZA: Hello there.",
        "audio_lines": "crowd murmur",
    }]}
    out = sw._normalize(data)
    sc = out["scenes"][0]
    check("dialogue preserved", sc.get("dialogue") == "ELIZA: Hello there.",
          repr(sc.get("dialogue")))
    check("characters_present preserved",
          sc.get("characters_present") == ["ELIZA", "ANNA"],
          repr(sc.get("characters_present")))
    out2 = sw._normalize({"scenes": [{"id": 1, "image_prompt": "i",
                                       "video_prompt": "v"}]})
    check("missing dialogue normalized to None",
          out2["scenes"][0].get("dialogue") is None)


def test_narrate_dialogue_text_and_prompt_guard() -> None:
    section("narrate: per-line speaker strip + genre/prompt guard")
    from narrate import _dialogue_text, _is_prompt_text
    dlg = ("ELIZA: (Smiling) Oh! What a night.\n"
           "ANNA: Eliza, that's uncalled for.\n"
           "ELIZA: Uncalled for? It's just comedy!")
    out = _dialogue_text(dlg)
    check("every speaker prefix stripped, lines joined",
          out == "(Smiling) Oh! What a night. Eliza, that's uncalled for. "
                 "Uncalled for? It's just comedy!",
          repr(out))
    cfg = {"story": {"genre": "this girl standing still in the banquet hall"}}
    check("prompt guard flags genre-like text",
          _is_prompt_text("this girl standing still in the banquet hall and "
                          "making fun of her friends", cfg) is True)
    check("prompt guard passes real dialogue",
          _is_prompt_text("Oh! What a lovely evening.", cfg) is False)


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
        lib.library = []  # isolate from the real characters library
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
#  director: 1-second chunked rendering (high FPS + high res without hangs)
# --------------------------------------------------------------------------- #
def test_chunk_plan_and_chunk_scene_values() -> None:
    section("director: chunk plan + chunk scene values")
    import director as director_mod
    cfg = base_config()
    cfg["story"]["seconds_per_scene"] = 8
    cfg["render"]["chunked"] = True
    cfg["render"]["chunk_seconds"] = 1.0
    d = director_mod.Director(cfg)
    d.chars = type("FakeLib", (), {"enabled": False})()

    check("8s scene -> 8 chunks", d._chunk_plan(8.0) == 8, str(d._chunk_plan(8.0)))
    check("2.5s scene -> 3 chunks", d._chunk_plan(2.5) == 3, str(d._chunk_plan(2.5)))
    check("1s scene -> 1 chunk", d._chunk_plan(1.0) == 1, str(d._chunk_plan(1.0)))

    vals = d._scene_values({"image_prompt": "x", "video_prompt": "y"}, 0,
                           duration=1.0, chunk_idx=2, chunk_total=8)
    check("chunk duration in values", vals["video_duration"] == 1.0,
          str(vals["video_duration"]))
    check("chunk save prefix",
          vals["save_prefix"] == "director/scene_00_c02",
          vals["save_prefix"])

    full = d._scene_values({"image_prompt": "x", "video_prompt": "y"}, 0)
    check("full scene duration from config", full["video_duration"] == 8.0,
          str(full["video_duration"]))
    check("full scene prefix", full["save_prefix"] == "director/scene_00",
          full["save_prefix"])


def test_apply_chain_image() -> None:
    section("director: frame-chaining workflow rewiring")
    import director as director_mod
    cfg = base_config()
    d = director_mod.Director(cfg)

    class FakeComfy:
        def upload_image(self, path):
            return {"name": "chain.png", "subfolder": "", "type": "input"}

    d.comfy = FakeComfy()
    wf = json.loads(json.dumps(d.template))  # deep copy of the real template
    out = d._apply_chain_image(wf, "dummy.png")

    load_nodes = [n for n in out.values() if n.get("class_type") == "LoadImage"]
    check("LoadImage node added", len(load_nodes) == 1, repr(load_nodes))
    check("LoadImage filename set",
          bool(load_nodes) and load_nodes[0]["inputs"]["image"] == "chain.png",
          repr(load_nodes))
    # The video first-frame chain (node 26) now points at the LoadImage node.
    rewired = out["26"]["inputs"]["input"]
    check("node 26 rewired away from keyframe",
          rewired[0] != "11" and str(rewired[0]).isdigit(), repr(rewired))
    # The T2I keyframe subgraph (nodes 1..11) is dropped for continuation chunks.
    check("T2I keyframe subgraph removed",
          all(str(k) not in out for k in range(1, 12)),
          "nodes 1-11 still present")


def test_extract_last_frame() -> None:
    section("director: extract exact last frame of a chunk")
    import director as director_mod
    import numpy as np
    import imageio.v2 as imageio
    cfg = base_config()
    d = director_mod.Director(cfg)
    tmp = tempfile.mkdtemp(prefix="chunk_last_")
    try:
        vid = os.path.join(tmp, "c.mp4")
        frames = [np.full((64, 64, 3), i * 40, dtype=np.uint8) for i in range(5)]
        imageio.mimsave(vid, frames, fps=5)
        d.output_dir = tmp
        png = d._extract_last_frame(vid, 0, 0)
        check("last frame png produced", png and os.path.exists(png), repr(png))
        if png:
            last = imageio.imread(png)
            check("png holds the final frame (value 160)",
                  abs(int(np.mean(last)) - 160) <= 5,
                  str(int(np.mean(last))))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_render_scene_chunked() -> None:
    section("director: chunked render orchestration (3s scene -> 3x1s)")
    import director as director_mod
    cfg = base_config()
    cfg["story"]["seconds_per_scene"] = 3
    cfg["render"]["chunked"] = True
    cfg["render"]["chunk_seconds"] = 1.0
    cfg["render"]["chain_frames"] = True
    cfg["qc"]["enabled"] = True
    d = director_mod.Director(cfg)
    d.chars = type("FakeLib", (), {"enabled": False})()

    calls = []

    def fake_segment(scene, scene_idx, duration, chunk_idx, chunk_total,
                     chain_image):
        calls.append((duration, chunk_idx, chunk_total, chain_image))
        return (f"output/scene_{scene_idx:02d}_c{chunk_idx:02d}.mp4", 1,
                QCResult(passed=True, metrics={"motion_std": 1.0}),
                ("ip", "vp"))

    def fake_extract(path, scene_idx, chunk_idx):
        return (f"output/_chunk_frames/"
                f"scene_{scene_idx:02d}_c{chunk_idx:02d}_last.png")

    def fake_stitch(scene_idx, chunk_paths):
        return f"output/scene_{scene_idx:02d}.mp4"

    d._render_segment = fake_segment
    d._extract_last_frame = fake_extract
    d._stitch_chunks = fake_stitch

    scene = {"image_prompt": "p", "video_prompt": "v"}
    path, attempts, qc_result, locked = d.render_scene(scene, 2)

    check("3 segments requested", len(calls) == 3, repr(calls))
    check("each segment is 1s", all(c[0] == 1.0 for c in calls),
          repr([c[0] for c in calls]))
    check("chunk indices 0..2", [c[1] for c in calls] == [0, 1, 2],
          repr([c[1] for c in calls]))
    check("first segment starts from keyframe (no chain)",
          calls[0][3] is None)
    check("later segments chain from previous last frame",
          calls[1][3] is not None and calls[2][3] is not None)
    check("stitched scene path returned",
          path == "output/scene_02.mp4", repr(path))
    check("attempts summed across segments", attempts == 3, str(attempts))
    check("aggregate qc marks chunked",
          qc_result.metrics.get("chunked") is True,
          repr(qc_result.metrics))
    check("aggregate qc passed (all segments pass)",
          qc_result.passed is True, repr(qc_result.reasons))


def test_render_scene_chunked_fallback_to_single() -> None:
    section("director: chunking off -> single-shot render")
    import director as director_mod
    cfg = base_config()
    cfg["story"]["seconds_per_scene"] = 8
    cfg["render"]["chunked"] = False
    d = director_mod.Director(cfg)
    d.chars = type("FakeLib", (), {"enabled": False})()

    calls = []

    def fake_segment(scene, scene_idx, duration, chunk_idx, chunk_total,
                     chain_image):
        calls.append((duration, chunk_idx, chunk_total, chain_image))
        return (f"output/scene_{scene_idx:02d}.mp4", 1,
                QCResult(passed=True, metrics={"motion_std": 1.0}),
                ("ip", "vp"))

    d._render_segment = fake_segment
    d._stitch_chunks = lambda scene_idx, chunk_paths: ""

    path, attempts, qc_result, locked = d.render_scene(
        {"image_prompt": "p", "video_prompt": "v"}, 0)

    check("single segment render", len(calls) == 1, repr(calls))
    check("segment carries full scene duration", calls[0][0] == 8.0,
          str(calls[0][0]))
    check("no chunk index", calls[0][1] is None, repr(calls[0][1]))
    check("result is the single clip",
          path == "output/scene_00.mp4", repr(path))
    check("no chunk aggregate", qc_result is None or not
          qc_result.metrics.get("chunked"))


def test_effective_scene_seconds_video_length() -> None:
    section("director: total video length distributes across scenes")
    import director as director_mod
    cfg = base_config()
    d = director_mod.Director(cfg)
    check("no video_length -> seconds_per_scene",
          d._effective_scene_seconds() == 5.0,
          str(d._effective_scene_seconds()))
    cfg["story"]["video_length"] = 60
    cfg["story"]["scenes"] = 2
    d2 = director_mod.Director(cfg)
    check("60s total / 2 scenes -> 30s per scene",
          d2._effective_scene_seconds() == 30.0,
          str(d2._effective_scene_seconds()))
    vals = d2._scene_values({"image_prompt": "x", "video_prompt": "y"}, 0)
    check("scene values carry distributed duration",
          vals["video_duration"] == 30.0, str(vals["video_duration"]))


def test_segment_cap_prevents_oom() -> None:
    section("director: max_segment_seconds hard cap (OOM protection)")
    import director as director_mod
    cfg = base_config()
    cfg["story"]["seconds_per_scene"] = 60
    cfg["render"]["chunked"] = True
    cfg["render"]["chunk_seconds"] = 30.0       # user's broken setting
    cfg["render"]["max_segment_seconds"] = 1.0   # safety cap
    d = director_mod.Director(cfg)
    d.chars = type("FakeLib", (), {"enabled": False})()

    check("60s scene -> 60 segments when capped at 1s",
          d._chunk_plan(60.0) == 60, str(d._chunk_plan(60.0)))
    check("chunk_seconds=30 ignored: effective segment = 1s",
          d._segment_seconds() == 1.0, str(d._segment_seconds()))

    calls = []

    def fake_segment(scene, scene_idx, duration, chunk_idx, chunk_total,
                     chain_image):
        calls.append((duration, chunk_idx, chunk_total, chain_image))
        return (f"output/scene_{scene_idx:02d}_c{chunk_idx:02d}.mp4", 1,
                QCResult(passed=True, metrics={"motion_std": 1.0}),
                ("ip", "vp"))

    def fake_extract(path, scene_idx, chunk_idx):
        return (f"output/_chunk_frames/"
                f"scene_{scene_idx:02d}_c{chunk_idx:02d}_last.png")

    d._render_segment = fake_segment
    d._extract_last_frame = fake_extract
    d._stitch_chunks = lambda scene_idx, chunk_paths: \
        f"output/scene_{scene_idx:02d}.mp4"

    path, attempts, qc_result, locked = d.render_scene(
        {"image_prompt": "p", "video_prompt": "v"}, 0)

    check("60 segments rendered", len(calls) == 60, repr(len(calls)))
    check("no generation longer than the 1s cap",
          all(c[0] <= 1.0 + 1e-6 for c in calls),
          repr([c[0] for c in calls]))
    check("aggregate qc marks chunked",
          qc_result.metrics.get("chunked") is True,
          repr(qc_result.metrics))


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
        check("scene 2 (audio line only) NOT spoken",
              out[1] is None, repr(out[1]))
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
    test_custom_story_multiline_dialogue()
    test_normalize_preserves_dialogue()
    test_narrate_dialogue_text_and_prompt_guard()
    test_fallback_story_schema()
    test_helpers()
    test_extract_characters()
    test_ensure_locks_offline()
    test_build_ref_workflow()
    test_scene_blocks_and_voice()
    test_consistency_report()
    test_director_scene_values()
    test_chunk_plan_and_chunk_scene_values()
    test_apply_chain_image()
    test_extract_last_frame()
    test_render_scene_chunked()
    test_render_scene_chunked_fallback_to_single()
    test_effective_scene_seconds_video_length()
    test_segment_cap_prevents_oom()
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
