"""ARC-SORT video where each track's ID is replaced by "GERMANY 22" / "JAPAN 14"
once its jersey number has been read reliably.

Stages (cached in the output dir):
  1. ocr     - PARSeq reads on torso windows of every tracked box -> ocr_reads.npz
  2. lock    - per track: team from side of the net, number from agreeing reads
  3. render  - 1080p video with the labels switching at the lock frame

Usage:
  venv/bin/python number_video.py --video input/rally_4258.mp4 --video-hr input/rally_4258_1080.mp4 --out output/rally_4258
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from compare import encode_h264, id_color, load_frames_npz, read_frames
from jersey_ocr import JerseyReader

# Candidate number windows inside a person box: (left, right, top, bottom) as box fractions.
WINDOWS = [(.15, .85, .15, .45), (.22, .78, .18, .42), (.28, .72, .20, .40), (.20, .80, .25, .50)]
TEAM_LEFT, TEAM_RIGHT = "JAPAN", "GERMANY"
TEAM_COLOR = {"JAPAN": (40, 30, 220), "GERMANY": (20, 200, 255)}  # BGR: red / gold


# ------------------------------------------------------------------- net
def net_positions(video_path, start_x, anchors, y=400.0):
    """Net x per frame (720p coords): camera motion carries the point, anchors remove drift."""
    frames = read_frames(video_path)
    prev = cv2.cvtColor(next(frames), cv2.COLOR_BGR2GRAY)
    pt, xs = np.array([start_x, y, 1.0]), [start_x]
    for frame in frames:
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(prev, 1500, 0.01, 8)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, p0, None)
        A, _ = cv2.estimateAffinePartial2D(p0[st == 1], p1[st == 1], method=cv2.RANSAC,
                                           ransacReprojThreshold=2)
        pt = np.array([*(A @ pt), 1.0])
        xs.append(pt[0])
        prev = g
    xs = np.array(xs)
    idx = np.array(sorted(anchors))
    resid = np.array([anchors[i] - xs[i] for i in idx])
    return xs + np.interp(np.arange(len(xs)), idx, resid)


# ------------------------------------------------------------------- ocr
def is_referee(frame, box):
    """Officials wear blue shirts; skip them so they never get a team number."""
    x1, y1, x2, y2 = box
    h = y2 - y1
    torso = frame[int(y1 + .15 * h):int(y1 + .45 * h), int(x1):int(x2)]
    if torso.size == 0:
        return False
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    blue = (hsv[..., 0] > 95) & (hsv[..., 0] < 125) & (hsv[..., 1] > 90) & (hsv[..., 2] > 60)
    return blue.mean() > 0.35


def run_ocr(video_hr, tracks, scale, step):
    """Return rows (frame, track_id, number or -1, conf, referee_flag)."""
    reader = JerseyReader()
    rows = []
    for fi, frame in enumerate(tqdm(read_frames(video_hr), total=len(tracks), desc="ocr")):
        if fi % step:
            continue
        for *box, tid in tracks[fi]:
            x1, y1, x2, y2 = [v * scale for v in box]
            w, h = x2 - x1, y2 - y1
            ref = is_referee(frame, (x1, y1, x2, y2))
            crops = []
            for l, r, t, b in WINDOWS:
                c = frame[max(int(y1 + t * h), 0):int(y1 + b * h), max(int(x1 + l * w), 0):int(x1 + r * w)]
                if c.shape[0] > 6 and c.shape[1] > 6:
                    crops.append(c)
            reads = [x for x in reader.read(crops) if x[0] is not None]
            if reads:
                num, conf = max(reads, key=lambda x: x[1])
                rows.append((fi, int(tid), int(num), conf, ref))
            else:
                rows.append((fi, int(tid), -1, 0.0, ref))
    return np.array(rows, dtype=np.float64)


# ------------------------------------------------------------------ lock
FALSE_FRIENDS = {1, 7}  # arms and shirt stripes read as 1 / 7, so these need more votes


def lock_numbers(reads, tracks, net_x, min_conf, min_votes, window):
    """Per track: team by majority side of the net, and a number timeline.

    A number is taken (or replaces the current one, e.g. after a tracker ID switch)
    once `min_votes` reads with conf >= min_conf agree within `window` frames and
    form the majority there. Returns {tid: {"team", "referee", "events": [(frame, number)],
    "support": [frames of reads agreeing with the label in force]}}.
    """
    side_votes = {}
    for fi, ft in enumerate(tracks):
        for x1, _, x2, _, tid in ft:
            side_votes.setdefault(int(tid), []).append((x1 + x2) / 2 < net_x[fi])

    info = {}
    for tid, votes in side_votes.items():
        team = TEAM_LEFT if np.mean(votes) >= 0.5 else TEAM_RIGHT
        r = reads[reads[:, 1] == tid]
        if len(r) and r[:, 4].mean() > 0.5:
            info[tid] = {"team": None, "referee": True, "events": [], "support": []}
            continue
        good = r[(r[:, 2] >= 0) & (r[:, 3] >= min_conf)]
        label, events, support = None, [], []
        for fi, num in good[:, [0, 2]].astype(int):
            recent = good[(good[:, 0] > fi - window) & (good[:, 0] <= fi)]
            top, n = Counter(recent[:, 2].astype(int)).most_common(1)[0]
            need = min_votes + 2 if top in FALSE_FRIENDS else min_votes
            if label is not None and (str(top) in str(label) or str(label) in str(top)):
                need = 2 * min_votes  # 18 -> 8, 21 -> 2: usually a digit hidden by an arm or a turn
            if top != label and top == num and n >= need and n > len(recent) / 2:
                label = int(top)
                events.append((int(fi), label))
            if num == label:
                support.append(int(fi))
        info[tid] = {"team": team, "referee": False, "events": events, "support": support}
    return info


def labels_per_frame(info, tracks):
    """Per frame {tid: (number, changed_at)}, dropping duplicates of the same team+number in
    one frame (the track with the most recent agreeing read keeps the label)."""
    out = []
    for fi, ft in enumerate(tracks):
        cand = {}
        for tid in ft[:, 4].astype(int):
            d = info.get(int(tid))
            ev = [e for e in d["events"] if e[0] <= fi] if d else []
            if ev:
                last = max([s for s in d["support"] if s <= fi], default=ev[-1][0])
                cand[int(tid)] = (ev[-1][1], ev[-1][0], last)
        best = {}
        for tid, (num, _, last) in cand.items():
            key = (info[tid]["team"], num)
            if key not in best or last > cand[best[key]][2]:
                best[key] = tid
        out.append({tid: cand[tid][:2] for tid in best.values()})
    return out


# ---------------------------------------------------------------- render
def label_box(img, p1, text, bg, fg, scale=0.8):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, 2)
    cv2.rectangle(img, (p1[0], p1[1] - th - 10), (p1[0] + tw + 10, p1[1]), bg, -1)
    cv2.putText(img, text, (p1[0] + 5, p1[1] - 5), cv2.FONT_HERSHEY_DUPLEX, scale, fg, 2, cv2.LINE_AA)


def render(video_hr, tracks, info, scale, fps, out_path):
    W, H = 1920, 1080
    labels = labels_per_frame(info, tracks)
    raw = out_path.with_suffix(".raw.mp4")
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for fi, frame in enumerate(tqdm(read_frames(video_hr), total=len(tracks), desc="render")):
        for *box, tid in tracks[fi]:
            tid = int(tid)
            x1, y1, x2, y2 = [int(v * scale) for v in box]
            if tid in labels[fi]:
                team = info[tid]["team"]
                number, changed = labels[fi][tid]
                c = TEAM_COLOR[team]
                just = fi - changed < int(fps * 0.6)  # brief flash at the moment of the read
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255) if just else c, 5 if just else 3)
                label_box(frame, (x1, y1), f"{team} {number}", c,
                          (0, 0, 0) if team == "GERMANY" else (255, 255, 255))
            else:
                c = id_color(tid)
                cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
                label_box(frame, (x1, y1), f"ID {tid}", c, (0, 0, 0), 0.65)

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (W, 70), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        cv2.putText(frame, "ARC-SORT + jersey OCR", (24, 48), cv2.FONT_HERSHEY_DUPLEX, 1.3,
                    (255, 255, 255), 2, cv2.LINE_AA)
        txt = f"identified on screen: {len(labels[fi])}/{len(tracks[fi])}"
        (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)
        cv2.putText(frame, txt, (W - tw - 24, 46), cv2.FONT_HERSHEY_DUPLEX, 1.0,
                    (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(frame)
    writer.release()
    encode_h264(raw, out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="input/rally_4258.mp4")
    ap.add_argument("--video-hr", default="input/rally_4258_1080.mp4")
    ap.add_argument("--out", default="output/rally_4258")
    ap.add_argument("--step", type=int, default=2, help="run OCR every Nth frame")
    ap.add_argument("--min-conf", type=float, default=0.8)
    ap.add_argument("--min-votes", type=int, default=3)
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--net-anchors", default="0:780,150:840,300:690,450:670,609:460",
                    help="frame:x pairs of the net post in 720p coords, read off the frames by eye "
                         "(defaults are for the VNL rally in the README)")
    args = ap.parse_args()

    out = Path(args.out)
    tracks = load_frames_npz(out / "tracks_arcsort.npz")
    scale, fps = 1080 / 720, 25.0

    net_path = out / "net_x.npy"
    if not net_path.exists():
        anchors = {int(f): float(x) for f, x in (p.split(":") for p in args.net_anchors.split(","))}
        np.save(net_path, net_positions(args.video, anchors[min(anchors)], anchors))
    net_x = np.load(net_path)

    ocr_path = out / "ocr_reads.npy"
    if not ocr_path.exists():
        np.save(ocr_path, run_ocr(args.video_hr, tracks, scale, args.step))
    reads = np.load(ocr_path)

    info = lock_numbers(reads, tracks, net_x, args.min_conf, args.min_votes, args.window)
    (out / "jersey_ids.json").write_text(json.dumps({str(k): v for k, v in info.items()}, indent=1))
    for tid, d in sorted(info.items()):
        print(f"track {tid:3d}: " + ("referee" if d["referee"] else
              f"{d['team']:8s} " + (" -> ".join(f"#{n}@{f}" for f, n in d["events"]) or "no number")))

    render(args.video_hr, tracks, info, scale, fps, out / "arcsort_jersey_ids.mp4")
    print(f"written {out / 'arcsort_jersey_ids.mp4'}")


if __name__ == "__main__":
    main()
