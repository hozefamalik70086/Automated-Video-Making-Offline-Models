#!/usr/bin/env python3
"""video_score.py — analyze and score a generated video (0-100 "film score").

Runs on the SAME signals the director's QC already uses (motion, frozen,
black frames) but expands them into a multi-dimensional 0-100 score with a
per-metric breakdown so you can see exactly WHY a clip scored well or poorly:

  motion       — how much visible motion (frame-diff, or real optical flow)
  consistency  — temporal stability / low flicker
  frozen       — share of frozen/identical frames
  black        — share of black/blank frames
  sharpness    — focus / clarity (gradient variance)
  exposure     — brightness level + stability

Composite = weighted average -> 0-100 + grade (A..F).

Engine: numpy + imageio (already installed). If opencv (cv2) is importable it
upgrades motion to real Farneback optical flow and sharpness to Laplacian
variance. Everything else degrades gracefully to numpy-only.

Usage:
    python video_score.py <video.mp4> [--json] [--expected SECONDS]
    python video_score.py --dir <output_dir> [--json]   # score every clip +
                                                        # final film, writes
                                                        # <dir>/video_scores.json
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

_BLACK_LUM = 12.0      # frame mean below this -> "black" (matches qc.py)
_FROZEN_DIFF = 0.6     # numpy frame diff below this -> "frozen" (matches qc.py)

# --------------------------------------------------------------------------- #
#  reading
# --------------------------------------------------------------------------- #
def _probe(path: str):
    import imageio
    with imageio.get_reader(path, "ffmpeg") as rdr:
        meta = rdr.get_meta_data()
        fps = float(meta.get("fps", 0) or 0)
        duration = float(meta.get("duration", 0) or 0)
        frames = int(rdr.count_frames())
        size = meta.get("size", (0, 0)) or (0, 0)
        return {"fps": fps, "duration": duration, "frames": frames,
                "width": int(size[0] or 0), "height": int(size[1] or 0)}


def _read_frames(path: str, max_samples: int = 128):
    """Yield up to max_samples RGB float32 frames (0-255)."""
    import imageio
    with imageio.get_reader(path, "ffmpeg") as rdr:
        total = max(int(rdr.count_frames()), 0)
        step = max(1, total // max_samples) if total else 1
        for i, frame in enumerate(rdr):
            if i % step != 0:
                continue
            arr = np.asarray(frame, dtype=np.float32)
            yield arr


# --------------------------------------------------------------------------- #
#  metric extraction (numpy engine)
# --------------------------------------------------------------------------- #
def _metrics_numpy(frames):
    """Compute motion/black/frozen/luma/sharpness from frame list (0-255)."""
    diffs, lumas, sharp = [], [], []
    prev = None
    black = 0
    total = 0
    for f in frames:
        total += 1
        gray = f.mean(axis=2) if f.ndim == 3 else f
        lumas.append(float(gray.mean()))
        if gray.mean() < _BLACK_LUM:
            black += 1
        # sharpness ~ variance of the gradient (blur -> low)
        if gray.size:
            gy = gray[1:, :] - gray[:-1, :]
            gx = gray[:, 1:] - gray[:, :-1]
            s = 0.5 * (float(np.var(gy)) + float(np.var(gx))) if gy.size and gx.size else 0.0
            sharp.append(s)
        if prev is not None:
            d = float(np.mean(np.abs(gray - prev)))
            if d > 1e-6:
                diffs.append(d)
            else:
                diffs.append(0.0)
        prev = gray
    if not diffs:
        diffs = [0.0]
    black_pct = 100.0 * black / max(total, 1)
    frozen_pct = 100.0 * sum(1 for d in diffs if d < _FROZEN_DIFF) / len(diffs)
    mean_motion = float(np.mean(diffs))
    motion_std = float(np.std(diffs))
    luma_mean = float(np.mean(lumas)) if lumas else 0.0
    luma_std = float(np.std(lumas)) if lumas else 0.0
    sharpness = float(np.mean(sharp)) if sharp else 0.0
    # flicker index: how bursty/inconsistent the frame-to-frame change is
    flicker = float(np.std(diffs) / (np.mean(diffs) + 1e-6))
    return {
        "mean_motion": mean_motion, "motion_std": motion_std,
        "frozen_pct": frozen_pct, "black_pct": black_pct,
        "luma_mean": luma_mean, "luma_std": luma_std,
        "sharpness": sharpness, "flicker": flicker,
        "engine": "numpy",
    }


# --------------------------------------------------------------------------- #
#  metric extraction (opencv engine — optional upgrade)
# --------------------------------------------------------------------------- #
def _metrics_cv2(frames):
    import cv2  # type: ignore  # optional
    flows, lumas, lapl = [], [], []
    prev_g = None
    black = 0
    total = 0
    for f in frames:
        total += 1
        gray = cv2.cvtColor(f.astype("uint8"), cv2.COLOR_RGB2GRAY)
        lumas.append(float(gray.mean()))
        if gray.mean() < _BLACK_LUM:
            black += 1
        lapl.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        if prev_g is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_g, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            flows.append(float(mag.mean()))
        prev_g = gray
    if not flows:
        flows = [0.0]
    black_pct = 100.0 * black / max(total, 1)
    # a frame is "frozen" when there is effectively no optical flow
    frozen_pct = 100.0 * sum(1 for m in flows if m < 0.15) / len(flows)
    mean_motion = float(np.mean(flows))
    motion_std = float(np.std(flows))
    luma_mean = float(np.mean(lumas)) if lumas else 0.0
    luma_std = float(np.std(lumas)) if lumas else 0.0
    sharpness = float(np.mean(lapl)) if lapl else 0.0
    flicker = float(np.std(flows) / (np.mean(flows) + 1e-6))
    return {
        "mean_motion": mean_motion, "motion_std": motion_std,
        "frozen_pct": frozen_pct, "black_pct": black_pct,
        "luma_mean": luma_mean, "luma_std": luma_std,
        "sharpness": sharpness, "flicker": flicker,
        "engine": "opencv",
    }


# --------------------------------------------------------------------------- #
#  scoring (0-100 per metric)
# --------------------------------------------------------------------------- #
def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def score_metrics(m: dict) -> dict:
    """Turn raw metrics into 0-100 per-metric scores."""
    eng = m.get("engine", "numpy")

    # motion: calibrate to observed LTX clips
    if eng == "opencv":
        m_floor, m_ceil = 0.2, 3.5
    else:
        m_floor, m_ceil = 0.8, 6.0
    motion = 100.0 * _clamp((m["mean_motion"] - m_floor) / (m_ceil - m_floor), 0, 1)

    # consistency: penalize luma instability + bursty motion (flicker)
    luma_score = 100.0 * float(np.exp(-(m["luma_std"] / 5.0) ** 2))
    flicker_score = 100.0 * float(np.exp(-(m["flicker"] / 1.2) ** 2))
    consistency = 0.5 * luma_score + 0.5 * flicker_score

    frozen = 100.0 * (1.0 - _clamp(m["frozen_pct"] / 25.0, 0, 1))
    black = 100.0 * (1.0 - _clamp(m["black_pct"] / 15.0, 0, 1))

    # sharpness: soft curve on gradient variance / Laplacian variance
    if eng == "opencv":
        s_ref = 60.0      # Laplacian var reference for 640x360 content
    else:
        s_ref = 120.0     # gradient-variance reference (0-255 frames)
    sharpness = 100.0 * (1.0 - float(np.exp(-m["sharpness"] / s_ref)))

    # exposure: level + stability
    lm = m["luma_mean"]
    if lm < 40:
        level = 100.0 * _clamp(lm / 40.0, 0, 1)
    elif lm > 220:
        level = 100.0 * _clamp((255 - lm) / 35.0, 0, 1)
    else:
        level = 100.0
    stability = 100.0 * float(np.exp(-(m["luma_std"] / 5.0) ** 2))
    exposure = 0.6 * level + 0.4 * stability

    return {
        "motion": round(motion, 1),
        "consistency": round(consistency, 1),
        "frozen": round(frozen, 1),
        "black": round(black, 1),
        "sharpness": round(sharpness, 1),
        "exposure": round(exposure, 1),
    }


_WEIGHTS = {"motion": 0.30, "consistency": 0.15, "frozen": 0.15,
            "black": 0.10, "sharpness": 0.15, "exposure": 0.15}


def grade_for(score: float) -> str:
    if score >= 85:
        return "A · Excellent"
    if score >= 70:
        return "B · Good"
    if score >= 55:
        return "C · Fair"
    if score >= 40:
        return "D · Poor"
    return "F · Bad"


# --------------------------------------------------------------------------- #
#  main entry
# --------------------------------------------------------------------------- #
def score_video(path: str, expected: float | None = None,
                prompt: str | None = None) -> dict:
    """Score one video file. Returns a full report dict (never raises)."""
    try:
        probe = _probe(path)
        frames = list(_read_frames(path))
        if not frames:
            return {"file": path, "error": "no readable frames"}

        try:
            import cv2  # noqa: F401
            metrics = _metrics_cv2(frames)
        except Exception:
            metrics = _metrics_numpy(frames)

        scores = score_metrics(metrics)
        film_score = round(sum(_WEIGHTS[k] * scores[k] for k in _WEIGHTS), 1)

        # optional prompt-alignment (only if torch + open_clip installed)
        prompt_score = None
        if prompt:
            try:
                prompt_score = round(_prompt_alignment(path, prompt), 2)
            except Exception:
                prompt_score = None

        result = {
            "file": os.path.basename(path),
            "probe": {k: (round(v, 3) if isinstance(v, float) else v)
                      for k, v in probe.items()},
            "metrics": {k: round(v, 3) if isinstance(v, float) else v
                        for k, v in metrics.items()},
            "scores": scores,
            "film_score": film_score,
            "grade": grade_for(film_score),
            "engine": metrics["engine"],
            "prompt_score": prompt_score,
        }
        if expected:
            ok = probe["duration"] >= expected * 0.9
            result["duration_ok"] = bool(ok)
        return result
    except Exception as exc:  # noqa: BLE001
        return {"file": os.path.basename(path), "error": str(exc)}


def _prompt_alignment(path: str, prompt: str) -> float:
    """Optional CLIP text<->frames alignment (needs torch + open_clip)."""
    import torch  # noqa: F401
    import open_clip  # type: ignore

    frames = list(_read_frames(path, max_samples=8))
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k")
    tok = open_clip.get_tokenizer("ViT-B-32")
    text = open_clip.tokenize([prompt])
    sims = []
    with torch.no_grad():
        for f in frames:
            img = preprocess(f.astype("uint8")).unsqueeze(0)
            feats = model.encode_image(img)
            feats /= feats.norm(dim=-1, keepdim=True)
            te = model.encode_text(text)
            te /= te.norm(dim=-1, keepdim=True)
            sims.append(float((feats @ te.T).item()))
    return float(np.mean(sims)) * 100.0


def score_dir(out_dir: str) -> dict:
    """Score every scene_*.mp4 + final_film*.mp4 in out_dir."""
    files = sorted(glob.glob(os.path.join(out_dir, "scene_*.mp4"))) + \
        sorted(glob.glob(os.path.join(out_dir, "final_film*.mp4")))
    results = {}
    for f in files:
        r = score_video(f)
        results[os.path.basename(f)] = r
    # persist
    out = os.path.join(out_dir, "video_scores.json")
    try:
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(results, fh, ensure_ascii=False, indent=2)
    except Exception:
        pass
    return results


def _print_table(results: dict) -> None:
    print(f"\n{'file':<24}{'score':>7}  {'grade':<14}{'motion':>7}"
          f"{'cons':>6}{'frz':>6}{'blk':>6}{'sharp':>7}{'exp':>7}")
    for name, r in results.items():
        if "error" in r:
            print(f"{name:<24}  ERROR {r['error']}")
            continue
        s = r["scores"]
        print(f"{name:<24}{r['film_score']:>7.1f}  {r['grade']:<14}"
              f"{s['motion']:>7.1f}{s['consistency']:>6.1f}{s['frozen']:>6.1f}"
              f"{s['black']:>6.1f}{s['sharpness']:>7.1f}{s['exposure']:>7.1f}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Score a generated video (0-100)")
    ap.add_argument("video", nargs="?", help="video file to score")
    ap.add_argument("--dir", default=None, help="score all clips in a folder")
    ap.add_argument("--json", action="store_true", help="emit raw JSON")
    ap.add_argument("--expected", type=float, default=None,
                    help="expected duration in seconds")
    ap.add_argument("--prompt", default=None,
                    help="optional text prompt for CLIP alignment (needs torch)")
    args = ap.parse_args(argv)

    if args.dir:
        results = score_dir(args.dir)
        if args.json:
            print(json.dumps(results, ensure_ascii=False, indent=2))
        else:
            _print_table(results)
        print(f"\n-> wrote {os.path.join(args.dir, 'video_scores.json')}")
        return

    if not args.video:
        ap.error("provide a video path or --dir")
    r = score_video(args.video, expected=args.expected, prompt=args.prompt)
    if args.json:
        print(json.dumps(r, ensure_ascii=False, indent=2))
    else:
        if "error" in r:
            print(f"ERROR: {r['error']}")
        else:
            _print_table({r["file"]: r})


if __name__ == "__main__":
    sys.exit(main())
