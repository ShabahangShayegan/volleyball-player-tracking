"""ByteTrack vs ARC-SORT on the same detections, rendered as comparison videos.

Stages (each cached in the output dir, so re-runs skip finished work):
  1. detect  - YOLO person detections for every frame  -> detections.npz
  2. track   - ByteTrack and ARC-SORT on identical dets -> tracks_<name>.npz
  3. render  - per-tracker videos + side-by-side + stacked comparison

Usage:
  venv/bin/python compare.py input/clip.mp4 --out output/
"""

import argparse
import colorsys
import subprocess
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

TRACKERS = ("bytetrack", "arcsort")
LABELS = {"bytetrack": "ByteTrack", "arcsort": "ARC-SORT"}


def read_frames(video_path):
    cap = cv2.VideoCapture(str(video_path))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield frame
    cap.release()


def video_info(video_path):
    cap = cv2.VideoCapture(str(video_path))
    info = dict(
        fps=cap.get(cv2.CAP_PROP_FPS),
        n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    cap.release()
    return info


# ---------------------------------------------------------------- 1. detect
def detect(video_path, out_path, model_name, imgsz, conf):
    from ultralytics import YOLO

    model = YOLO(model_name)
    info = video_info(video_path)
    per_frame = []
    for frame in tqdm(read_frames(video_path), total=info["n"], desc="detect"):
        r = model.predict(frame, imgsz=imgsz, conf=conf, classes=[0], verbose=False)[0]
        b = r.boxes
        dets = np.concatenate(
            [b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None], b.cls.cpu().numpy()[:, None]],
            axis=1,
        ) if len(b) else np.empty((0, 6))
        per_frame.append(dets.astype(np.float32))
    np.savez_compressed(out_path, *per_frame)


def load_frames_npz(path):
    data = np.load(path)
    return [data[f"arr_{i}"] for i in range(len(data.files))]


# ----------------------------------------------------------------- 2. track
def run_bytetrack(video_path, dets_per_frame, fps):
    import supervision as sv
    from trackers import ByteTrackTracker

    tracker = ByteTrackTracker(frame_rate=fps, lost_track_buffer=30)
    out = []
    for frame, dets in tqdm(zip(read_frames(video_path), dets_per_frame),
                            total=len(dets_per_frame), desc="bytetrack"):
        sv_dets = sv.Detections(
            xyxy=dets[:, :4].astype(np.float32),
            confidence=dets[:, 4].astype(np.float32),
            class_id=dets[:, 5].astype(int),
        )
        res = tracker.update(sv_dets, frame)
        keep = res.tracker_id >= 0 if len(res) else np.zeros(0, bool)
        out.append(np.concatenate([res.xyxy[keep], res.tracker_id[keep, None]], axis=1)
                   if keep.any() else np.empty((0, 5)))
    return out


def run_arcsort(video_path, dets_per_frame, fps):
    from arcsort import ARCTracker

    tracker = ARCTracker.from_preset("sportsmot")  # motion-only (no ReID weights)
    out = []
    for frame, dets in tqdm(zip(read_frames(video_path), dets_per_frame),
                            total=len(dets_per_frame), desc="arcsort"):
        out.append(tracker.update(dets.astype(np.float64), frame))
    return out


# ---------------------------------------------------------------- 3. render
def id_color(track_id):
    h = (int(track_id) * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def track_stats(tracks):
    lengths = {}
    for frame_tracks in tracks:
        for tid in frame_tracks[:, 4].astype(int):
            lengths[tid] = lengths.get(tid, 0) + 1
    return lengths


def draw(frame, tracks, name, seen_ids):
    img = frame.copy()
    for x1, y1, x2, y2, tid in tracks:
        tid = int(tid)
        seen_ids.add(tid)
        c = id_color(tid)
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(img, p1, p2, c, 2)
        label = str(tid)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(img, (p1[0], p1[1] - th - 6), (p1[0] + tw + 6, p1[1]), c, -1)
        cv2.putText(img, label, (p1[0] + 3, p1[1] - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 2, cv2.LINE_AA)

    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (img.shape[1], 56), (0, 0, 0), -1)
    img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
    cv2.putText(img, LABELS[name], (16, 40), cv2.FONT_HERSHEY_DUPLEX, 1.1,
                (255, 255, 255), 2, cv2.LINE_AA)
    counter = f"unique IDs so far: {len(seen_ids)}"
    (tw, _), _ = cv2.getTextSize(counter, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)
    cv2.putText(img, counter, (img.shape[1] - tw - 16, 38), cv2.FONT_HERSHEY_DUPLEX,
                0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def encode_h264(raw_path, final_path):
    """Re-encode OpenCV's mp4v output to H.264 so LinkedIn/phones play it."""
    import imageio_ffmpeg

    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
                    "-i", str(raw_path), "-c:v", "libx264", "-preset", "slow", "-crf", "20",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(final_path)],
                   check=True)
    raw_path.unlink()


def render(video_path, tracks_by_name, out_dir, fps):
    info = video_info(video_path)
    w, h = info["w"], info["h"]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    sizes = {
        "bytetrack": (w, h),
        "arcsort": (w, h),
        "side_by_side": (w * 2, h),
        "stacked": (w, h * 2),
    }
    writers = {k: cv2.VideoWriter(str(out_dir / f"{k}.raw.mp4"), fourcc, fps, s)
               for k, s in sizes.items()}
    seen = {n: set() for n in TRACKERS}
    for i, frame in enumerate(tqdm(read_frames(video_path), total=info["n"], desc="render")):
        panels = {n: draw(frame, tracks_by_name[n][i], n, seen[n]) for n in TRACKERS}
        writers["bytetrack"].write(panels["bytetrack"])
        writers["arcsort"].write(panels["arcsort"])
        writers["side_by_side"].write(np.hstack([panels["bytetrack"], panels["arcsort"]]))
        writers["stacked"].write(np.vstack([panels["bytetrack"], panels["arcsort"]]))
    for k, wr in writers.items():
        wr.release()
        encode_h264(out_dir / f"{k}.raw.mp4", out_dir / f"{k}.mp4")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", default="output")
    ap.add_argument("--model", default="yolo11l.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.05)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    fps = video_info(args.video)["fps"]

    det_path = out_dir / "detections.npz"
    if not det_path.exists():
        detect(args.video, det_path, args.model, args.imgsz, args.conf)
    dets = load_frames_npz(det_path)

    runners = {"bytetrack": run_bytetrack, "arcsort": run_arcsort}
    tracks = {}
    for name in TRACKERS:
        path = out_dir / f"tracks_{name}.npz"
        if not path.exists():
            np.savez_compressed(path, *runners[name](args.video, dets, fps))
        tracks[name] = load_frames_npz(path)

    print(f"\n{len(dets)} frames, same {args.model} detections for both trackers")
    for name in TRACKERS:
        lengths = track_stats(tracks[name])
        short = sum(1 for v in lengths.values() if v < fps)  # tracks shorter than 1 s
        print(f"{LABELS[name]:>10}: {len(lengths):4d} unique IDs, "
              f"{short:3d} lasting < 1 s, median length {int(np.median(list(lengths.values())))} frames")

    render(args.video, tracks, out_dir, fps)
    print(f"videos written to {out_dir}/")


if __name__ == "__main__":
    main()
