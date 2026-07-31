#!/usr/bin/env python3
"""narrate.py — optional narration / subtitle post-processor.

Runs AFTER director.py has produced output/final_film.mp4 + report.json.
For every scene it:

  1. synthesizes a voice-over from the scene's ``audio_lines`` (edge-tts),
  2. places each clip's narration at the start of its scene,
  3. optionally burns subtitles (ASS) using the audio_lines text,
  4. muxes the narration track into the film -> final_film_narrated.mp4.

It is fully optional and defensive: every failure is caught and printed as a
warning, and it never touches the base pipeline's files.

Usage:
    python narrate.py [--config config.json] [--dir output]

Requirements (optional):
    pip install edge-tts          # only needed when audio.enabled is true
The script already uses imageio-ffmpeg's bundled ffmpeg binary.
"""

import argparse
import json
import os
import sys
import time
import traceback

# --------------------------------------------------------------------------- #
#  Paths
# --------------------------------------------------------------------------- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
#  ffmpeg helpers
# --------------------------------------------------------------------------- #
def get_ffmpeg() -> str:
    """Return a usable ffmpeg executable (imageio-ffmpeg bundled, else PATH)."""
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if os.path.exists(exe):
            return exe
    except Exception:  # noqa: BLE001
        pass
    return "ffmpeg"


def run_ffmpeg(args: list) -> bool:
    import subprocess
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=1800)
        if proc.returncode != 0:
            print("  [narrate] ffmpeg stderr tail: " +
                  (proc.stderr or "")[-600:].replace("\n", " "))
            return False
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  [narrate] ffmpeg failed: {exc}")
        return False


# --------------------------------------------------------------------------- #
#  Narration synthesis
# --------------------------------------------------------------------------- #
def make_narration(report: dict, cfg: dict, out_dir: str) -> list:
    """Synthesize one mp3 per scene (or None if unavailable/empty).

    Returns a list parallel to report["scenes"] of mp3 paths or None.
    """
    audio = cfg.get("audio", {})
    engine = audio.get("engine", "edge-tts")
    voice = audio.get("voice", "en-US-GuyNeural")
    rate = audio.get("rate", "+0%")

    try:
        import edge_tts  # type: ignore
    except Exception as exc:  # noqa: BLE001
        print(f"  [narrate] edge-tts not installed, skipping voice: {exc}")
        return [None] * len(report.get("scenes", []))

    os.makedirs(os.path.join(out_dir, "_narration"), exist_ok=True)
    results = []
    for i, sc in enumerate(report.get("scenes", []), start=1):
        text = (sc.get("audio_lines") or "").strip()
        if not text:
            results.append(None)
            continue
        out = os.path.join(out_dir, "_narration", f"scene_{i:02d}.mp3")
        try:
            print(f"  [narrate] synth scene {i}: {text[:70]}…")
            if engine != "edge-tts":
                results.append(None)
                continue
            comm = edge_tts.Communicate(text, voice=voice, rate=rate)
            asyncio_sync(comm, out)
            results.append(out if os.path.exists(out) and os.path.getsize(out) > 0 else None)
        except Exception as exc:  # noqa: BLE001
            print(f"  [narrate] synth scene {i} failed: {exc}")
            results.append(None)
    return results


def asyncio_sync(communicate, out_path: str) -> None:
    """Run an edge_tts.Communicate.save() to completion."""
    import asyncio
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(communicate.save(out_path))
    finally:
        loop.close()


# --------------------------------------------------------------------------- #
#  Track building + muxing
# --------------------------------------------------------------------------- #
def build_narration_track(narration: list, scenes: list, out_dir: str) -> str:
    """Concatenate per-scene narration into one audio track.

    Each scene's narration is delayed to the scene's start time and mixed.
    Scenes without narration get silence so timings stay aligned.
    """
    if not any(n for n in narration):
        return ""

    dur = 0.0
    starts = []
    for sc in scenes:
        starts.append(dur)
        dur += float(sc.get("duration_seconds") or 5.0)

    inputs = []
    filters = []
    labels = []
    for i, (n, t) in enumerate(zip(narration, starts)):
        if not n:
            continue
        inputs.append("-i")
        inputs.append(n)
        delay = int(t * 1000)
        labels.append(f"n{i}")
        filters.append(f"[{len(inputs)//2-1}:a]adelay={delay}|{delay}[n{i}]")

    if not inputs:
        return ""

    # measure total narration length so we can pad silence for the film tail
    amix = "".join(f"[{l}]" for l in labels) + f"amix=inputs={len(labels)}:normalize=0[aout]"
    track = os.path.join(out_dir, "_narration", "track.wav")
    ok = run_ffmpeg(
        [get_ffmpeg(), "-y"] + inputs +
        ["-filter_complex", ";".join(filters + [amix]),
         "-ac", "2", "-ar", "44100", track]
    )
    return track if ok else ""


def build_subtitles(scenes: list, out_dir: str) -> str:
    """Write an ASS subtitle file for the narration lines."""
    lines = [(sc.get("audio_lines") or "").strip() for sc in scenes]
    if not any(lines):
        return ""
    dur = 0.0
    offsets = []
    for sc in scenes:
        offsets.append(dur)
        dur += float(sc.get("duration_seconds") or 5.0)

    ass = os.path.join(out_dir, "_narration", "subs.ass")
    with open(ass, "w", encoding="utf-8") as f:
        f.write("[Script Info]\nScriptType: v4.00+\nPlayResX: 640\nPlayResY: 360\n"
                "WrapStyle: 0\nScaledBorderAndShadow: yes\n\n")
        f.write("[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, "
                "SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
                "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
                "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, "
                "MarginV, Encoding\n")
        f.write("Style: Default,Arial,22,&H00FFFFFF,&H000000FF,&H00000000,"
                "&H80000000,-1,0,0,0,100,100,0,0,1,2,0,2,24,24,40,1\n\n")
        f.write("[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, "
                "MarginR, MarginV, Effect, Text\n")
        for i, txt in enumerate(lines):
            if not txt:
                continue
            t0 = offsets[i]
            t1 = t0 + float(scenes[i].get("duration_seconds") or 5.0)
            start = fmt_ass(t0)
            end = fmt_ass(t1)
            safe = txt.replace("\n", " ").replace("{", "｛").replace("}", "｝")
            f.write(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{safe}\n")
    return ass


def fmt_ass(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds - int(seconds)) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


# --------------------------------------------------------------------------- #
#  Orchestration
# --------------------------------------------------------------------------- #
def narrate(cfg: dict, report: dict, out_dir: str) -> str:
    """Produce final_film_narrated.mp4. Returns its path or ''."""
    audio = cfg.get("audio", {})
    if not audio.get("enabled"):
        print("[narrate] audio.enabled is false — nothing to do.")
        return ""

    film = report.get("final_film", "")
    if not film:
        print("[narrate] report has no final_film — nothing to narrate.")
        return ""
    film_path = os.path.join(BASE_DIR, film) if not os.path.isabs(film) else film
    if not os.path.exists(film_path):
        print(f"[narrate] film not found: {film_path}")
        return ""

    scenes = report.get("scenes", [])
    narration = make_narration(report, cfg, out_dir)
    track = build_narration_track(narration, scenes, out_dir)
    subs = build_subtitles(scenes, out_dir) if audio.get("subtitles", True) else ""

    out = os.path.join(out_dir, "final_film_narrated.mp4")
    cmd = [get_ffmpeg(), "-y", "-i", film_path]
    if track:
        cmd += ["-i", track]
    if subs:
        cmd += ["-i", subs]
    cmd += ["-map", "0:v:0"]
    if track:
        cmd += ["-map", "1:a:0"]
    if subs:
        cmd += ["-map", "2:s:0"]
    if subs:
        # burn the subtitles into the video
        cmd += ["-vf", f"ass={subs}"]
    cmd += ["-c:v", "libx264", "-crf", "20", "-preset", "fast"]
    if track:
        cmd += ["-c:a", "aac", "-b:a", "192k", "-shortest"]
    else:
        cmd += ["-an"]
    if subs and not track:
        # keep original audio if we only burned subs
        cmd[-2] = "-c:a"
        cmd[-1] = "aac"
        cmd += ["-b:a", "192k"]

    print("[narrate] muxing narration + subtitles → " + out)
    if not run_ffmpeg(cmd):
        print("[narrate] failed to produce narrated film — base film unchanged.")
        return ""
    if os.path.exists(out):
        print(f"[narrate] OK -> {os.path.relpath(out, BASE_DIR)}")
        return out
    return ""


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Narration / subtitle post-processor")
    ap.add_argument("--config", default="config.json",
                    help="config.json path (for the audio section)")
    ap.add_argument("--dir", default=None, help="output dir override")
    args = ap.parse_args(argv)

    t0 = time.time()
    try:
        cfg = load_json(os.path.join(BASE_DIR, args.config))
    except Exception as exc:  # noqa: BLE001
        print(f"[narrate] cannot load config: {exc}")
        return

    out_dir = args.dir or os.path.join(BASE_DIR,
                                       cfg.get("director", {}).get("output_dir", "output"))
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, "report.json")
    try:
        report = load_json(report_path)
    except Exception as exc:  # noqa: BLE001
        print(f"[narrate] cannot load {report_path}: {exc}")
        return

    try:
        narrate(cfg, report, out_dir)
        print(f"[narrate] done in {time.time()-t0:.1f}s")
    except Exception:  # noqa: BLE001
        print("[narrate] unexpected error — base pipeline untouched.")
        traceback.print_exc()


if __name__ == "__main__":
    sys.exit(main())
