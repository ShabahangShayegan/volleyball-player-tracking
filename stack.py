"""Stack the ByteTrack video on top of the ARC-SORT + jersey OCR video (1280x1440, phone-friendly).

Usage:
  venv/bin/python stack.py --out output/rally_4258
"""

import argparse
import subprocess
from pathlib import Path

import imageio_ffmpeg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output/rally_4258")
    args = ap.parse_args()

    out = Path(args.out)
    top, bottom = out / "bytetrack.mp4", out / "arcsort_jersey_ids.mp4"
    dst = out / "stacked_bytetrack_vs_arcsort_jersey.mp4"
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
                    "-i", str(top), "-i", str(bottom),
                    "-filter_complex", "[0:v]scale=1280:720[a];[1:v]scale=1280:720[b];[a][b]vstack=inputs=2",
                    "-c:v", "libx264", "-preset", "slow", "-crf", "20", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(dst)], check=True)
    print(f"written {dst}")


if __name__ == "__main__":
    main()
