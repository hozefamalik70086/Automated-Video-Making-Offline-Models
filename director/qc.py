"""
qc.py — Automatic Quality Control ("director reviews the 5s clip").

After ComfyUI renders a scene, the director inspects the actual video file
before accepting it:

  * duration        — must be close to the requested length (no truncated export)
  * black frames    — a mostly-black frame suggests a failed/empty generation
  * frozen frames   — frames identical to their predecessor (generation stalled)
  * motion          — overall motion std; too low means the clip is static

Each check is configurable in config.json under "qc". A clip FAILS if any
metric exceeds its threshold, and the director re-shoots it with new seeds
(up to qc.max_attempts).

Frames are sampled (not every frame read) so QC stays fast.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover - older imageio layout
    import imageio  # type: ignore

_BLACK_LUM = 12.0        # frame mean below this -> "black"
_FROZEN_DIFF = 0.6       # mean abs frame diff below this -> "frozen"


@dataclass
class QCResult:
    passed: bool
    metrics: dict = field(default_factory=dict)
    reasons: list = field(default_factory=list)

    def summary(self) -> str:
        parts = [f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
                 for k, v in self.metrics.items()]
        return " | ".join(parts)


class QualityChecker:
    def __init__(self, config: dict) -> None:
        self.cfg = config.get("qc", {})

    # ------------------------------------------------------------- probes
    def probe(self, path: str) -> Optional[dict]:
        """Open the video and return {fps, duration, frames, width, height}.
        Returns None if the file cannot be opened (hard failure)."""
        try:
            with imageio.get_reader(path, "ffmpeg") as rdr:
                meta = rdr.get_meta_data()
                fps = float(meta.get("fps", 0) or 0)
                duration = float(meta.get("duration", 0) or 0)
                # NOTE: meta['nframes'] is inf for the ffmpeg plugin; use the
                # reliable count_frames() instead (counts by decoding).
                frames = int(rdr.count_frames())
                size = meta.get("size", (0, 0)) or (0, 0)
                width = int(size[0] or 0)
                height = int(size[1] or 0)
            return {"fps": fps, "duration": duration, "frames": frames,
                    "width": width, "height": height}
        except Exception:
            return None

    def _sample_frames(self, path: str, max_samples: int = 96):
        """Yield up to max_samples frames as grayscale float arrays."""
        with imageio.get_reader(path, "ffmpeg") as rdr:
            total = max(int(rdr.count_frames()), 0)
            step = max(1, total // max_samples) if total else 1
            for i, frame in enumerate(rdr):
                if i % step != 0:
                    continue
                gray = np.asarray(frame, dtype=np.float32)
                if gray.ndim == 3:
                    gray = gray.mean(axis=2)
                yield gray

    # ------------------------------------------------------------- analyze
    def analyze(self, path: str, expected_duration: float) -> QCResult:
        cfg = self.cfg
        probe = self.probe(path)
        if probe is None:
            return QCResult(False, {"error": "unreadable"},
                            ["video file could not be opened/decoded"])

        metrics = {k: round(v, 3) for k, v in probe.items()}
        reasons: list[str] = []

        # duration check
        tol = float(cfg.get("duration_tolerance_seconds", 0.6))
        if abs(probe["duration"] - expected_duration) > tol:
            reasons.append(f"duration {probe['duration']:.2f}s != "
                           f"{expected_duration:.2f}s (±{tol})")
        if probe["duration"] < float(cfg.get("min_duration_seconds", 4.0)):
            reasons.append(f"duration {probe['duration']:.2f}s too short")
        if probe["frames"] < int(cfg.get("min_frames", 40)):
            reasons.append(f"only {probe['frames']} frames")

        # frame-level checks
        diffs: list[float] = []
        prev: Optional[np.ndarray] = None
        black = 0
        total = 0
        for frame in self._sample_frames(path):
            total += 1
            if frame.mean() < _BLACK_LUM:
                black += 1
            if prev is not None:
                diffs.append(float(np.mean(np.abs(frame - prev))))
            prev = frame

        if total == 0:
            return QCResult(False, metrics, reasons + ["no readable frames"])

        black_pct = 100.0 * black / total
        metrics["black_frames_pct"] = round(black_pct, 2)
        if black_pct > float(cfg.get("max_black_frames_pct", 15.0)):
            reasons.append(f"{black_pct:.1f}% black frames")

        if diffs:
            frozen = sum(1 for d in diffs if d < _FROZEN_DIFF)
            frozen_pct = 100.0 * frozen / len(diffs)
            motion_std = float(np.std(diffs))
            metrics["frozen_frames_pct"] = round(frozen_pct, 2)
            metrics["motion_std"] = round(motion_std, 3)
            if frozen_pct > float(cfg.get("max_frozen_frames_pct", 25.0)):
                reasons.append(f"{frozen_pct:.1f}% frozen frames")
            if motion_std < float(cfg.get("min_motion_std", 2.0)):
                reasons.append(f"motion_std {motion_std:.2f} too low (static clip)")

        return QCResult(passed=not reasons, metrics=metrics, reasons=reasons)
