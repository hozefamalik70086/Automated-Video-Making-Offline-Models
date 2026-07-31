"""Temporary monitor: waits for the director run to finish producing
final_film.mp4 + report.json in the output dir, printing scene progress."""
import os
import sys
import time

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
REPORT = os.path.join(BASE, "report.json")
FILM = os.path.join(BASE, "final_film.mp4")
TIMEOUT = 1740  # seconds


def scene_files():
    if not os.path.isdir(BASE):
        return []
    return sorted(
        f for f in os.listdir(BASE)
        if f.startswith("scene_") and f.endswith(".mp4")
    )


def main():
    baseline = set(scene_files())
    print("monitoring {} (pre-existing scenes: {})".format(
        BASE, sorted(baseline) or "none"))
    deadline = time.time() + TIMEOUT
    while time.time() < deadline:
        for f in scene_files():
            if f not in baseline:
                baseline.add(f)
                print("[{}] NEW scene file: {} ({} bytes)".format(
                    time.strftime("%H:%M:%S"), f,
                    os.path.getsize(os.path.join(BASE, f))))
        if os.path.exists(REPORT) and os.path.exists(FILM):
            print("[{}] DONE: report.json + final_film.mp4 present".format(
                time.strftime("%H:%M:%S")))
            print("scene files:", scene_files())
            print("film size:", os.path.getsize(FILM), "bytes")
            return 0
        time.sleep(15)
    print("TIMEOUT after {}s. scene files: {}".format(TIMEOUT, scene_files()))
    return 1


if __name__ == "__main__":
    sys.exit(main())
