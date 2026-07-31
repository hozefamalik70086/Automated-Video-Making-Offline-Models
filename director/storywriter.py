"""
storywriter.py — Generates the story that the Director will shoot.

Priority order (matching the user's requirement that custom input bypasses the
AI story writer and is optional):
  1. CUSTOM STORY (bypass):  if `config.story.custom_story_file` exists and has
     content, it is parsed into scenes directly. NO LLM is contacted.
     Format: scene blocks separated by a line of exactly `---`.
             Optional "IMAGE:" / "VIDEO:" prefixed lines; plain text is used
             for both. See custom_story.txt / README for the format.
  2. LLM STORY:               otherwise, if `config.llm.backend` is
     "ollama" or "openai", ask the model for a JSON story with N scenes.
  3. TEMPLATE FALLBACK:       if the LLM is unavailable or errors, a small
     built-in story is used so the pipeline still runs end-to-end.

Every scene has the shape:
    {"id": int, "image_prompt": str, "video_prompt": str,
     "audio_lines": str, "duration_seconds": float}
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Optional

import requests

SYSTEM_PROMPT = """You are the story writer for an AI film "director mode".
Write a short visual story as STRICT JSON only — no markdown, no commentary.
Schema:
{"story_title": string,
 "scenes": [
   {"id": 1,
    "image_prompt": "detailed text-to-image prompt for the opening frame (subject, lighting, camera angle, style)",
    "video_prompt": "text-to-video motion prompt that starts FROM the image_prompt scene (movement, action, camera motion, 5 seconds)",
    "audio_lines": "short on-screen/ambient line or sound description",
    "duration_seconds": 5},
   ...
 ]}
Rules:
- Each scene must be a new shot that continues the story.
- image_prompt and video_prompt must describe the SAME scene.
- Keep prompts clear, concrete, and cinematic; avoid copyrighted characters.
- VIDEO prompts are the most important: they MUST describe continuous, SPECIFIC,
  visible motion from the very first frame — a camera move (pan, orbit, dolly,
  tilt, handheld drift, push-in) PLUS moving subjects/elements (waves, wind,
  water, cloth, hair, clouds, dust, light flicker, vehicles, characters acting).
  Name concrete actions and directions, e.g. "The camera slowly orbits right as
  waves crash and spray the rocks, the keeper's coat whips in the wind and the
  lantern flame flickers hard."
- NEVER use static or slow words in video_prompt: avoid "still", "frozen",
  "slow", "gentle sway", "static", "calm". A slow push-in with no moving element
  generates a frozen video. Every frame must show change.
"""

FALLBACK_STORY = {
    "story_title": "The Lantern Keeper (template fallback)",
    "scenes": [
        {"id": 1,
         "image_prompt": ("A lone lighthouse keeper on a rocky cliff at dusk, "
                          "warm lantern light, cinematic wide shot, moody sky"),
         "video_prompt": ("The camera slowly orbits the keeper as waves crash "
                          "and spray the rocks, the keeper's coat whips in the "
                          "wind, the lantern flame flickers hard, drifting fog, "
                          "5 seconds"),
         "audio_lines": "Wind and crashing waves",
         "duration_seconds": 5},
        {"id": 2,
         "image_prompt": ("The keeper climbs the spiral stairs inside the "
                          "lighthouse, warm glow, cinematic medium shot"),
         "video_prompt": ("The camera tilts up and glides behind the keeper "
                          "climbing, the lantern swings with each step, dust "
                          "motes drift in the warm light, coat tails sway, "
                          "5 seconds"),
         "audio_lines": "Creaking wood, soft footsteps",
         "duration_seconds": 5},
        {"id": 3,
         "image_prompt": ("The keeper throws open the lamp room door; the great "
                          "lantern blazes, rain-streaked glass, cinematic"),
         "video_prompt": ("The great lantern ignites with a blinding flash and "
                          "roars, beams sweep across the sea, rain streaks down "
                          "the glass, steam and smoke billow, the keeper shields "
                          "his eyes, 5 seconds"),
         "audio_lines": "The roar of the lamp igniting",
         "duration_seconds": 5},
        {"id": 4,
         "image_prompt": ("A ship far out at sea turns toward the light, "
                          "silhouettes, golden beam cutting the night, cinematic"),
         "video_prompt": ("The camera drifts forward as the ship glides toward "
                          "the rotating beam, waves roll and sparkle, clouds "
                          "race across the moon, the beam sweeps past, 5 seconds"),
         "audio_lines": "Faint ship horn, sea breeze",
         "duration_seconds": 5},
    ],
}


class StoryWriter:
    def __init__(self, config: dict, base_dir: str) -> None:
        self.cfg = config
        self.base_dir = base_dir
        story_cfg = config.get("story", {})
        self.custom_file = story_cfg.get(
            "custom_story_file", "custom_story.txt")
        self.genre = story_cfg.get("genre", "cinematic")
        self.num_scenes = int(story_cfg.get("scenes", 4))
        self.seconds = float(story_cfg.get("seconds_per_scene", 5))

    # ------------------------------------------------------------- entry
    def _finalize(self, story: dict) -> dict:
        """Honor the dashboard's scenes / seconds_per_scene settings.

        Caps the scene list to `num_scenes` (cycling the source story if it
        has fewer scenes) and forces EVERY scene's duration to the global
        `seconds_per_scene`, so the config the user sets in the dashboard is
        authoritative regardless of what the LLM or fallback template emits.
        """
        src = (story or {}).get("scenes", []) or []
        scenes = []
        n = max(1, int(self.num_scenes))
        for i in range(n):
            s = src[i % len(src)] if src else {}
            scenes.append({
                "id": i + 1,
                "image_prompt": str(s.get("image_prompt", "")).strip(),
                "video_prompt": str(s.get("video_prompt", "")).strip(),
                "audio_lines": str(s.get("audio_lines", "")).strip(),
                "duration_seconds": float(self.seconds),
            })
        return {"story_title": str(story.get("story_title", "story")),
                "scenes": scenes}

    def get_story(self) -> dict:
        custom = self._read_custom()
        if custom is not None:
            print("[story] using CUSTOM story (LLM bypassed)")
            return self._finalize(self._parse_custom(custom))
        if self.cfg.get("llm", {}).get("backend", "none") != "none":
            try:
                story = self._ask_llm()
                if story:
                    print(f"[story] LLM wrote: {story.get('story_title', '')}")
                    return self._finalize(story)
                print("[story] LLM returned unusable output; using fallback")
            except Exception as e:  # noqa: BLE001 - fall back gracefully
                print(f"[story] LLM unavailable ({e}); using fallback")
        else:
            print("[story] llm.backend == 'none'; using template fallback")
        return self._finalize(FALLBACK_STORY)

    # ------------------------------------------------------------- custom
    def _read_custom(self) -> Optional[str]:
        path = os.path.join(self.base_dir, self.custom_file)
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()
        # strip comment lines (start with '#' after leading whitespace) so the
        # template header in custom_story.txt is never mistaken for story text
        content = "\n".join(
            ln for ln in lines if not ln.strip().startswith("#")
        ).strip()
        return content if content else None

    def _parse_custom(self, text: str) -> dict:
        text = "\n".join(
            ln for ln in text.splitlines() if not ln.strip().startswith("#")
        ).strip()
        blocks = [b.strip() for b in re.split(r"^\s*-{3,}\s*$", text, flags=re.M)
                  if b.strip()]
        if not blocks:
            blocks = [text]
        scenes = []
        for idx, block in enumerate(blocks, start=1):
            image_prompt, video_prompt = None, None
            lines = [ln for ln in block.splitlines() if ln.strip()]
            rest = []
            for ln in lines:
                low = ln.strip().lower()
                if low.startswith("image:"):
                    image_prompt = ln.split(":", 1)[1].strip()
                elif low.startswith("video:"):
                    video_prompt = ln.split(":", 1)[1].strip()
                else:
                    rest.append(ln.strip())
            body = " ".join(rest).strip()
            if image_prompt is None:
                image_prompt = body
            if video_prompt is None:
                video_prompt = body
            scenes.append({
                "id": idx,
                "image_prompt": image_prompt,
                "video_prompt": video_prompt,
                "audio_lines": "",
                "duration_seconds": self.seconds,
            })
        return {"story_title": "Custom story", "scenes": scenes}

    # ------------------------------------------------------------- llm
    def _ask_llm(self) -> Optional[dict]:
        llm = self.cfg.get("llm", {})
        backend = llm.get("backend", "none")
        attempts = max(1, int(llm.get("max_attempts", 5)))
        user_prompt = self._build_user_prompt()
        for attempt in range(1, attempts + 1):
            try:
                if backend == "ollama":
                    self._ollama_warmup(llm)
                    content = self._ollama_chat(llm, user_prompt)
                elif backend == "openai":
                    content = self._openai_chat(llm, user_prompt)
                else:
                    return None
                parsed = self._parse_llm_json(content)
                if parsed and parsed.get("scenes"):
                    return parsed
                print(f"[story] LLM attempt {attempt}/{attempts} returned "
                      "empty/invalid output; retrying...")
            except Exception as exc:  # noqa: BLE001
                print(f"[story] LLM attempt {attempt}/{attempts} error "
                      f"({exc}); retrying...")
            time.sleep(1)
        return None

    def _build_user_prompt(self) -> str:
        user_prompt = (
            f"Write a {self.num_scenes}-scene {self.genre} story. "
            f"Each scene is exactly {self.seconds} seconds long. "
            "Every scene must include an image_prompt (for a still keyframe) "
            "and a video_prompt (dynamic, continuous visible motion from the "
            f"very first frame, {self.seconds} seconds). "
        )
        char = str(self.cfg.get("story", {}).get("character", "") or "").strip()
        if char:
            user_prompt += (
                f"Use the SAME main character '{char}' in every scene. "
            )
        user_prompt += (
            "Return ONLY the strict JSON object: "
            '{"story_title": "...", "scenes": [{"id":1, '
            '"image_prompt": "...", "video_prompt": "...", '
            '"audio_lines": "...", "duration_seconds": '
            f'{int(self.seconds)}}}, ...]}}'
        )
        return user_prompt

    def _build_system_prompt(self) -> str:
        """SYSTEM_PROMPT plus an optional character-consistency instruction."""
        prompt = SYSTEM_PROMPT
        char = str(self.cfg.get("story", {}).get("character", "") or "").strip()
        if char:
            prompt += (
                "\nCharacter consistency (REQUIRED): describe the SAME main "
                f"character in every scene: '{char}'. Keep their appearance "
                "(clothes, body, key features) identical across all "
                "image_prompt and video_prompt entries.\n"
            )
        return prompt

    def _ollama_warmup(self, llm: dict) -> None:
        """Tiny call to make sure the model is loaded so the real call does
        not race the first-load path (which tends to return empty on qwen3)."""
        try:
            url = (llm.get("ollama_url", "http://127.0.0.1:11434").rstrip("/")
                   + "/api/generate")
            requests.post(
                url,
                json={"model": llm.get("ollama_model", "gemma3:12b"),
                      "prompt": "ping", "stream": False,
                      "keep_alive": llm.get("keep_alive", "30m")},
                timeout=120,
            )
        except Exception:  # noqa: BLE001
            pass

    def _ollama_chat(self, llm: dict, user_prompt: str) -> str:
        url = (llm.get("ollama_url", "http://127.0.0.1:11434").rstrip("/")
               + "/api/chat")
        payload = {
            "model": llm.get("ollama_model", "gemma3:12b"),
            "messages": [{"role": "system",
                          "content": self._build_system_prompt()},
                         {"role": "user", "content": user_prompt}],
            "stream": False,
            "format": "json",
            "keep_alive": llm.get("keep_alive", "30m"),
            "options": {
                "temperature": float(llm.get("temperature", 0.8)),
                "num_predict": int(llm.get("max_tokens", 2048)),
            },
        }
        r = requests.post(url, json=payload, timeout=int(llm.get("timeout", 300)))
        r.raise_for_status()
        return r.json().get("message", {}).get("content", "")

    def _openai_chat(self, llm: dict, user_prompt: str) -> str:
        base = llm.get("openai_base_url", "http://127.0.0.1:1234/v1").rstrip("/")
        url = base + "/chat/completions"
        headers = {"Authorization":
                   f"Bearer {llm.get('openai_api_key', 'not-needed')}"}
        payload = {
            "model": llm.get("openai_model", "qwen3-14b"),
            "messages": [{"role": "system",
                          "content": self._build_system_prompt()},
                         {"role": "user", "content": user_prompt}],
            "temperature": float(llm.get("temperature", 0.8)),
            "max_tokens": int(llm.get("max_tokens", 2048)),
            "response_format": {"type": "json_object"},
        }
        r = requests.post(url, json=payload, headers=headers,
                          timeout=int(llm.get("timeout", 300)))
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def _parse_llm_json(self, content: str) -> Optional[dict]:
        content = (content or "").strip()
        try:
            return self._normalize(json.loads(content))
        except json.JSONDecodeError:
            pass
        m = re.search(r"\{.*\}", content, flags=re.S)
        if m:
            try:
                return self._normalize(json.loads(m.group(0)))
            except json.JSONDecodeError:
                pass
        return None

    def _normalize(self, data: dict) -> dict:
        scenes = []
        for i, s in enumerate(data.get("scenes", []), start=1):
            scenes.append({
                "id": int(s.get("id", i)),
                "image_prompt": str(s.get("image_prompt", "")).strip(),
                "video_prompt": str(s.get("video_prompt", "")).strip(),
                "audio_lines": str(s.get("audio_lines", "")).strip(),
                "duration_seconds": float(s.get("duration_seconds", self.seconds)),
            })
        return {"story_title": str(data.get("story_title", "LLM story")),
                "scenes": scenes}

    @staticmethod
    def _clone(story: dict) -> dict:
        return json.loads(json.dumps(story))
