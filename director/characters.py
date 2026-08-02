#!/usr/bin/env python3
"""characters.py — Character-locked cast library for the Director pipeline.

Implements the "Character-Locked Script-to-Video" phases on the OFFLINE stack
(no cloud APIs — Ollama for text, ComfyUI Z-Image-Turbo for images, edge-tts
for voices):

    Phase 0  extract_characters()        story -> ordered cast
                                         (names + speaking flags from
                                         characters_present / dialogue)
    Phase 1  ensure_locks()              Ollama writes a rich visual_descriptor;
                                         ComfyUI renders a master reference
                                         image (Z-Image-Turbo T2I); both are
                                         stored in a lock file. Prompt-locking
                                         feeds every scene prompt today; the
                                         reference PNG is kept so an IP-Adapter
                                         can be wired in later.
    Phase 2  voice locks                 speaking characters get a locked
                                         edge-tts voice (engine: edge-tts,
                                         per-character, stored in the lock).
    Phase 3  scene_character_blocks()    per-scene descriptor + consistency
                                         anchor for the director to inject.
    Phase 4  write_consistency_report()  coverage + prompt-anchor checks +
                                         optional face-embedding check (only if
                                         a face library is installed).

Locks live in <base_dir>/characters/characters.json + characters/refs/.
A returning character (matched by normalized name) reuses its existing lock,
so identity stays stable across runs and stories.

Usage:
    python characters.py --config config.json --list
    python characters.py --config config.json --setup [--no-refs] [--force-descriptors]
    python characters.py --config config.json --consistency <output_dir>
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from typing import Optional

import requests

from comfy_api import ComfyUI

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Nodes of workflow_scene_template.json that form the T2I (Z-Image-Turbo)
# keyframe subgraph. A reference workflow = these nodes + a SaveImage.
REF_NODE_IDS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]
REF_SAVE_NODE = "60"

CONSISTENCY_ANCHOR = (
    "All characters appear exactly as described: identical face, hair, skin, "
    "clothing and accessories in every scene. No alteration to any "
    "character's appearance."
)

# edge-tts voices chosen per (gender, age) bucket. Child voices are limited on
# edge-tts, so the closest match is used; users can edit the lock JSON to swap.
EDGE_TTS_VOICES = {
    "child_female": "en-US-AnaNeural",
    "child_male": "en-US-AndrewNeural",
    "young_female": "en-US-AriaNeural",
    "young_male": "en-US-GuyNeural",
    "adult_female": "en-US-JennyNeural",
    "adult_male": "en-US-BrianNeural",
    "elder_female": "en-US-MichelleNeural",
    "elder_male": "en-US-ChristopherNeural",
}


def load_json(rel: str) -> dict:
    path = os.path.join(BASE_DIR, rel)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _norm(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "").strip().lower())


def _dialogue_speaker(dialogue: str) -> Optional[str]:
    """Extract the speaker from 'Name: line' / 'Name| line', else None."""
    text = (dialogue or "").strip()
    if not text:
        return None
    m = re.match(r"^\s*([^:|]{1,40})[|:]\s*", text)
    if not m:
        return None
    name = m.group(1).strip()
    if len(name) < 2:
        return None
    if any(k in name.lower() for k in ("camera", "narrator", "sound", "sfx",
                                       "close-up", "wide", "shot", "music")):
        return None
    return name


def _pick_edge_voice(voice_descriptor: str, name: str = "") -> str:
    low = ((voice_descriptor or "") + " " + (name or "")).lower()
    female = any(k in low for k in ("female", "woman", "girl", "mother",
                                    "grandmother", "her", "she"))
    child = any(k in low for k in ("child", "little girl", "little boy", "kid",
                                   "toddler", "boy", "girl", "6-year",
                                   "six-year", "small"))
    elder = any(k in low for k in ("elder", "elderly", "grandmother",
                                   "grandfather", "old woman", "old man",
                                   "senior", "wrinkled"))
    young = any(k in low for k in ("young", "teen", "teenager", "youth"))
    if elder:
        key = "elder_female" if female else "elder_male"
    elif child:
        key = "child_female" if female else "child_male"
    elif female:
        key = "young_female" if young else "adult_female"
    else:
        key = "young_male" if young else "adult_male"
    return EDGE_TTS_VOICES.get(key, "en-US-GuyNeural")


def _fallback_visual(name: str) -> str:
    # Stable, name-derived anchor so fallback (no-LLM) locks still differ from
    # each other. Real locks use a rich Ollama visual_descriptor instead; this
    # only runs when the LLM is unavailable (e.g. bad model name / 404).
    seed = sum(ord(c) for c in name.lower())
    attrs = [
        "a memorable silhouette and distinctive coloring",
        "a bold color palette and clean styling",
        "an unmistakable look and strong visual identity",
        "a distinctive appearance with high contrast",
    ]
    attr = attrs[seed % len(attrs)]
    return (f"A clearly visible {name} with {attr}; recognizable in every "
            "scene; cinematic lighting; high detail.")


class CharacterLibrary:
    def __init__(self, config: dict, base_dir: str = BASE_DIR) -> None:
        self.cfg = config
        self.base_dir = base_dir
        cc = config.get("characters", {}) or {}
        self.enabled = bool(cc.get("enabled", True))
        lib_rel = cc.get("library_dir", "characters")
        self.lib_dir = os.path.join(base_dir, lib_rel)
        self.refs_dir = os.path.join(self.lib_dir, "refs")
        os.makedirs(self.refs_dir, exist_ok=True)
        self.library_path = os.path.join(self.lib_dir, "characters.json")
        self.library = self._load_library()
        # lazy ComfyUI client + template cache (attr name != method name so the
        # instance attribute never shadows the accessor)
        self._comfy: Optional[ComfyUI] = None
        self._template: Optional[dict] = None

    # ---------------------------------------------------------- persistence
    def _load_library(self) -> list:
        if os.path.exists(self.library_path):
            try:
                with open(self.library_path, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:  # noqa: BLE001
                return []
        return []

    def save(self) -> None:
        with open(self.library_path, "w", encoding="utf-8") as f:
            json.dump(self.library, f, ensure_ascii=False, indent=2)

    def lock_for(self, name: str) -> Optional[dict]:
        n = _norm(name)
        for lock in self.library:
            if _norm(lock.get("name", "")) == n:
                return lock
        return None

    def summary(self) -> list:
        out = []
        for l in self.library:
            out.append({
                "char_id": l.get("char_id"),
                "name": l.get("name"),
                "speaking": bool(l.get("speaking")),
                "master_ref_image": l.get("master_ref_image") or "",
                "voice": ((l.get("voice_lock") or {}).get("voice")
                          if l.get("speaking") else None),
            })
        return out

    # ---------------------------------------------------------- Phase 0
    def extract_characters(self, story: dict) -> list:
        """Collect the ordered cast from the story's scenes."""
        chars: dict = {}
        scenes = story.get("scenes", []) or []
        for i, sc in enumerate(scenes, start=1):
            for name in (sc.get("characters_present") or []):
                n = _norm(name)
                if not n:
                    continue
                c = chars.setdefault(n, {"name": str(name).strip(),
                                         "speaking": False,
                                         "scenes": set()})
                c["scenes"].add(i)
            speaker = _dialogue_speaker(sc.get("dialogue"))
            if speaker:
                n = _norm(speaker)
                if n in chars:
                    chars[n]["speaking"] = True
                else:
                    chars[n] = {"name": speaker.strip(), "speaking": True,
                                "scenes": {i}}
        cast = []
        for i, (n, c) in enumerate(chars.items(), start=1):
            cast.append({"id": f"char_{i:03d}", "name": c["name"],
                         "speaking": c["speaking"],
                         "scene_ids": sorted(c["scenes"])})
        return cast

    # ---------------------------------------------------------- Phase 1+2
    def ensure_locks(self, story: dict, generate_refs: bool = True,
                     force_descriptors: bool = False) -> list:
        """Make sure every story character has a lock: visual + voice
        descriptors (Ollama), optional master reference image (ComfyUI).
        Existing locks are reused by name. Returns the locks used."""
        if not self.enabled:
            return []
        cast = self.extract_characters(story)
        if not cast:
            return []
        made = []
        next_id = len(self.library) + 1
        for char in cast:
            lock = self.lock_for(char["name"])
            if lock is None:
                lock = self._new_lock(char, next_id)
                next_id += 1
                self.library.append(lock)
            if force_descriptors or not lock.get("visual_descriptor"):
                self._write_descriptors(lock, story)
                self.save()
            if (generate_refs and not lock.get("master_ref_image")
                    and self._comfy_alive()):
                self._make_master_ref(lock)
                self.save()
            made.append(lock)
        self.save()
        return made

    def _new_lock(self, char: dict, char_id: int) -> dict:
        return {
            "char_id": f"char_{char_id:03d}",
            "name": char["name"],
            "speaking": bool(char["speaking"]),
            "scene_ids": list(char.get("scene_ids", [])),
            "visual_descriptor": "",
            "voice_descriptor": "",
            "master_ref_image": "",
            "ref_seed": None,
            "ref_prompt": "",
            "voice_lock": None,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    def _write_descriptors(self, lock: dict, story: dict) -> None:
        name = lock["name"]
        desc = self._ask_descriptors(name, story)
        if desc and desc.get("visual_descriptor"):
            lock["visual_descriptor"] = desc["visual_descriptor"].strip()
            lock["voice_descriptor"] = (desc.get("voice_descriptor") or "").strip()
        else:
            lock["visual_descriptor"] = _fallback_visual(name)
            lock["voice_descriptor"] = ""
        if lock["speaking"]:
            voice = _pick_edge_voice(lock["voice_descriptor"], name)
            lock["voice_lock"] = {"engine": "edge-tts", "voice": voice,
                                  "rate": "+0%", "speaking": True}
        else:
            lock["voice_lock"] = None
        v = lock["voice_lock"]["voice"] if lock["voice_lock"] else "-"
        print(f"  [char] locked {lock['char_id']} {name!r} "
              f"speaking={lock['speaking']} voice={v}")

    def _ask_descriptors(self, name: str, story: dict) -> Optional[dict]:
        llm = self.cfg.get("llm", {})
        if llm.get("backend", "none") == "none":
            return None
        system = (
            "You are a character designer for a film. For the named character "
            "write: (1) visual_descriptor — a RICH, consistent description of "
            "face, hair, skin, body, clothing, accessories and any signature "
            "prop, detailed enough that an image model renders the exact same "
            "person/object in every shot; (2) voice_descriptor — gender, age, "
            "pitch, accent and speaking style. Return STRICT JSON only: "
            '{"visual_descriptor": "...", "voice_descriptor": "..."}'
        )
        scenes = (story.get("scenes") or [])
        sample = [str(s.get("video_prompt", ""))[:120]
                  for s in scenes[:6] if s.get("video_prompt")]
        user = (f"Character name: {name}\n"
                f"Story: {story.get('story_title', '')}\n"
                f"Scene samples: {json.dumps(sample, ensure_ascii=False)}\n"
                "Write the descriptors. Return only the JSON object.")
        attempts = max(1, int(llm.get("max_attempts", 3)))
        for attempt in range(1, attempts + 1):
            try:
                content = self._llm_chat(llm, system, user)
                data = self._parse_json(content)
                if data and data.get("visual_descriptor"):
                    return data
                print(f"  [char] descriptor attempt {attempt}/{attempts} "
                      "empty/invalid; retrying...")
            except Exception as exc:  # noqa: BLE001
                print(f"  [char] descriptor attempt {attempt}/{attempts} "
                      f"error ({exc}); retrying...")
            time.sleep(1)
        return None

    def _llm_chat(self, llm: dict, system: str, user: str) -> str:
        backend = llm.get("backend", "none")
        if backend == "ollama":
            url = (llm.get("ollama_url", "http://127.0.0.1:11434").rstrip("/")
                   + "/api/chat")
            payload = {
                "model": llm.get("ollama_model", "qwen3:4b"),
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}],
                "stream": False,
                "format": "json",
                "keep_alive": llm.get("keep_alive", "30m"),
                "options": {
                    "temperature": float(llm.get("temperature", 0.6)),
                    "num_predict": int(llm.get("max_tokens", 1024)),
                },
            }
            r = requests.post(url, json=payload, timeout=180)
            r.raise_for_status()
            return r.json().get("message", {}).get("content", "")
        if backend == "openai":
            base = (llm.get("openai_base_url", "http://127.0.0.1:1234/v1")
                    .rstrip("/"))
            r = requests.post(
                base + "/chat/completions",
                headers={"Authorization":
                         f"Bearer {llm.get('openai_api_key', 'not-needed')}"},
                json={"model": llm.get("openai_model", "qwen3-14b"),
                      "messages": [{"role": "system", "content": system},
                                   {"role": "user", "content": user}],
                      "temperature": float(llm.get("temperature", 0.6)),
                      "max_tokens": int(llm.get("max_tokens", 1024)),
                      "response_format": {"type": "json_object"}},
                timeout=180,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        return ""

    def _parse_json(self, content: str) -> Optional[dict]:
        content = (content or "").strip()
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content).strip()
        try:
            return json.loads(content)
        except Exception:  # noqa: BLE001
            m = re.search(r"\{.*\}", content, re.S)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:  # noqa: BLE001
                    return None
        return None

    # ---------------------------------------------------------- Phase 1.2
    def _get_comfy(self) -> Optional[ComfyUI]:
        if self._comfy is None:
            cu = self.cfg.get("comfyui", {})
            self._comfy = ComfyUI(
                url=cu.get("url", "http://127.0.0.1:8188"),
                timeout=float(cu.get("timeout_seconds", 30)),
            )
        return self._comfy

    def _comfy_alive(self) -> bool:
        try:
            c = self._get_comfy()
            return c is not None and c.is_alive()
        except Exception:  # noqa: BLE001
            return False

    def _load_template(self) -> Optional[dict]:
        if self._template is not None:
            return self._template
        rel = self.cfg.get("characters", {}).get(
            "ref_template", "workflow_scene_template.json")
        path = os.path.join(self.base_dir, rel)
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as f:
            self._template = json.load(f)
        return self._template

    def build_ref_workflow(self, prompt: str, seed: int, prefix: str) -> dict:
        """A minimal T2I-only workflow (Z-Image-Turbo subgraph + SaveImage)
        for rendering a master reference image from a character descriptor."""
        template = self._load_template()
        if not template:
            raise RuntimeError("scene template missing for ref workflow")
        wf = {}
        for nid in REF_NODE_IDS:
            if nid not in template:
                raise RuntimeError(f"template missing T2I node {nid}")
            wf[nid] = json.loads(json.dumps(template[nid]))
        wf["6"]["inputs"]["text"] = prompt
        wf["9"]["inputs"]["seed"] = int(seed)
        wf[REF_SAVE_NODE] = {"class_type": "SaveImage",
                             "inputs": {"images": ["11", 0],
                                        "filename_prefix": prefix}}
        return wf

    def _make_master_ref(self, lock: dict) -> None:
        name = lock["name"]
        prompt = (f"{lock['visual_descriptor']}, standing in a neutral pose, "
                  "studio lighting, full body, plain background, high resolution")
        seed = random.randint(0, 2**32 - 1)
        prefix = f"director/charref/{lock['char_id']}"
        try:
            wf = self.build_ref_workflow(prompt, seed, prefix)
            c = self._get_comfy()
            pid = c.submit(wf)
            timeout = float(self.cfg.get("comfyui", {}).get(
                "render_timeout_seconds", 3600))
            c.wait(pid, timeout=timeout)
            files = c.output_files(pid)
            target = None
            for f in files:
                if str(f.get("node_id")) == REF_SAVE_NODE and f.get("filename"):
                    target = f
                    break
            target = target or (files[0] if files else None)
            if not target:
                print(f"  [char] no output image for {name}")
                return
            dest = os.path.join(self.refs_dir, f"{lock['char_id']}_ref.png")
            c.download(target["filename"], dest, target.get("subfolder", ""),
                       target.get("type", "output"))
            lock["master_ref_image"] = os.path.relpath(dest, self.base_dir)
            lock["ref_seed"] = seed
            lock["ref_prompt"] = prompt
            print(f"  [char] master ref for {name} -> {lock['master_ref_image']}")
        except Exception as exc:  # noqa: BLE001
            print(f"  [char] ref generation failed for {name}: {exc}")

    # ---------------------------------------------------------- Phase 3
    def scene_character_blocks(self, scene: dict) -> tuple:
        """(image_blocks, video_note) for the scene's characters_present.

        image_blocks: full visual descriptors to append to the image prompt.
        video_note:   a short "same characters" note to append to video prompt.
        """
        image_blocks = []
        names = []
        for name in (scene.get("characters_present") or []):
            lock = self.lock_for(name)
            if lock and lock.get("visual_descriptor"):
                image_blocks.append(lock["visual_descriptor"])
                names.append(lock["name"])
            elif name:
                names.append(str(name).strip())
        video_note = ""
        if names:
            video_note = ("Characters present (identical appearance to the "
                          "keyframe image): " + ", ".join(names) + ".")
        return image_blocks, video_note

    def voice_for_scene(self, scene: dict) -> Optional[dict]:
        """Return the locked voice for the scene's dialogue speaker, else the
        first speaking character in the scene, else None."""
        speaker = _dialogue_speaker(scene.get("dialogue"))
        lock = self.lock_for(speaker) if speaker else None
        if lock and lock.get("voice_lock"):
            return lock["voice_lock"]
        for name in (scene.get("characters_present") or []):
            lock = self.lock_for(name)
            if lock and lock.get("voice_lock"):
                return lock["voice_lock"]
        return None

    # ---------------------------------------------------------- Phase 4
    def write_consistency_report(self, story: dict, results: list,
                                 out_dir: str) -> dict:
        """Cross-scene consistency review. Honest and dependency-light:
           * coverage   — which scenes each locked character should appear in
           * anchor     — did the locked descriptor actually reach the prompt
           * face check — embedding similarity ONLY if a face library exists
        """
        report = {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "method": ("prompt-locking (descriptor anchors) + scene coverage; "
                       "face-embedding check optional"),
            "face_embedding_check": self._face_check_status(),
            "characters": [],
        }
        by_scene: dict = {}
        for i, sc in enumerate((story.get("scenes") or []), start=1):
            by_scene[i] = set(_norm(n)
                              for n in (sc.get("characters_present") or []))
        for lock in self.library:
            scenes_present = sorted(
                i for i, names in by_scene.items()
                if _norm(lock["name"]) in names)
            anchor_ok = []
            probe = (lock.get("visual_descriptor") or "")[:50].lower()
            for r in results:
                idx = int(r.get("scene", 0))
                prompt = (r.get("image_prompt") or "").lower()
                anchor_ok.append({"scene": idx,
                                  "anchor_in_prompt":
                                      bool(probe) and probe in prompt})
            report["characters"].append({
                "char_id": lock["char_id"],
                "name": lock["name"],
                "speaking": lock["speaking"],
                "master_ref_image": lock.get("master_ref_image") or "",
                "scenes_present": scenes_present,
                "prompt_anchor_check": anchor_ok,
            })
        out = os.path.join(out_dir, "consistency_report.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"  [char] consistency report -> "
              f"{os.path.relpath(out, self.base_dir)}")
        return report

    def _face_check_status(self) -> str:
        for mod in ("face_recognition", "insightface", "facenet_pytorch"):
            try:
                __import__(mod)
                return (f"available ({mod}) — run a check to compare refs vs "
                        "scene keyframes")
            except Exception:  # noqa: BLE001
                continue
        return ("skipped — no face library installed (face_recognition / "
                "insightface / facenet_pytorch); install one to enable "
                "embedding-based cross-scene similarity")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Director character library")
    parser.add_argument("--config", default="config.json",
                        help="path to config.json (default: config.json)")
    parser.add_argument("--list", action="store_true",
                        help="show the locked cast")
    parser.add_argument("--setup", action="store_true",
                        help="extract chars, write descriptors, generate refs")
    parser.add_argument("--no-refs", action="store_true",
                        help="with --setup: skip ComfyUI master ref generation")
    parser.add_argument("--force-descriptors", action="store_true",
                        help="re-write descriptors even if a lock exists")
    parser.add_argument("--consistency", metavar="OUT_DIR",
                        help="write a consistency report for a finished run")
    args = parser.parse_args(argv)

    config = load_json(args.config)
    lib = CharacterLibrary(config, BASE_DIR)

    if args.list:
        print("\nLocked cast:")
        for l in lib.summary():
            print(f"  {l['char_id']}  {l['name']:<28} "
                  f"speaking={l['speaking']}  "
                  f"ref={l['master_ref_image'] or '-'}  "
                  f"voice={l['voice'] or '-'}")
        return

    if args.setup:
        from storywriter import StoryWriter
        story = StoryWriter(config, BASE_DIR).get_story()
        print(f"\n[char] story: {story.get('story_title')!r} "
              f"({len(story.get('scenes', []))} scenes)")
        cast = lib.extract_characters(story)
        cast_str = ", ".join(f"{c['name']}({c['speaking']})" for c in cast)
        print(f"[char] extracted cast: {cast_str}")
        lib.ensure_locks(story, generate_refs=not args.no_refs,
                         force_descriptors=args.force_descriptors)
        print(f"\n[char] library saved -> "
              f"{os.path.relpath(lib.library_path, BASE_DIR)}")
        return

    if args.consistency:
        out_dir = args.consistency
        results = []
        story = {"story_title": "", "scenes": []}
        report_path = os.path.join(out_dir, "report.json")
        if os.path.exists(report_path):
            with open(report_path, encoding="utf-8") as f:
                rep = json.load(f)
            results = rep.get("scenes", [])
            story = {"story_title": rep.get("story_title", ""),
                     "scenes": [{"characters_present":
                                 (r.get("characters_present") or []),
                                 "video_prompt": r.get("video_prompt", "")}
                                for r in results]}
        lib.write_consistency_report(story, results, out_dir)
        return

    parser.print_help()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except requests.RequestException as exc:
        print(f"\n[char] network error: {exc}")
