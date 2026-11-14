"""Offline validation against ground truth.

Examples
--------
  python -m rppg.evaluate synthetic
  python -m rppg.evaluate video my_clip.mp4 --gt my_oximeter.csv
  python -m rppg.evaluate ubfc  ~/data/UBFC-rPPG/DATASET_2
  python -m rppg.evaluate pure  ~/data/PURE
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .datasets import GroundTruth, find_pure_sequences, find_ubfc_subjects, load_auto, load_pure, load_ubfc
from .estimator import EstimatorConfig, analyze_hr
from .face import FaceROITracker
from .sources import pure_frames, video_frames
from .synthetic import SyntheticSpec, synthetic_frames


@dataclass
class Traces:
    t: np.ndarray
    rgb: np.ndarray  # NaN rows where no face
    motion: np.ndarray
    center_y: np.ndarray


def extract_traces(frames, tracker: FaceROITracker, max_seconds: float | None = None, label: str = "") -> Traces:
    t_l, rgb_l, mot_l, y_l = [], [], [], []
    for i, (t, frame) in enumerate(frames):
        if max_seconds and t > max_seconds:
            break
        obs = tracker.process(frame, t)
        t_l.append(t)
        rgb_l.append(obs.rgb if obs else np.full(3, np.nan))
        mot_l.append(obs.motion if obs else 0.0)
        y_l.append(obs.center_y if obs else np.nan)
        if i % 150 == 0:
            print(f"\r  {label} {t:6.1f}s", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    return Traces(np.array(t_l), np.array(rgb_l), np.array(mot_l), np.array(y_l))


def sliding_hr(tr: Traces, cfg: EstimatorConfig, step: float = 1.0) -> list[dict]:
    rows = []
    if len(tr.t) == 0:
        return rows
    for t_end in np.arange(tr.t[0] + cfg.hr_window, tr.t[-1] + 1e-9, step):
        m = (tr.t > t_end - cfg.hr_window) & (tr.t <= t_end)
        r = analyze_hr(tr.t[m], tr.rgb[m], tr.motion[m], cfg)
        rows.append({"t_end": round(float(t_end), 3), "status": r.status,
                     "est_bpm": r.value, "snr_db": r.snr_db})
    return rows


def metrics(est: np.ndarray, ref: np.ndarray) -> dict:
    if len(est) == 0:
        return {"n": 0}
    err = est - ref
    out = {
        "n": len(est),
        "MAE": float(np.abs(err).mean()),
        "RMSE": float(np.sqrt((err**2).mean())),
        "MAPE%": float((np.abs(err) / ref).mean() * 100),
        "pct<=5bpm": float((np.abs(err) <= 5).mean() * 100),  # % of windows within 5 BPM
        "bias": float(err.mean()),
        "LoA95": float(1.96 * err.std()),
    }
    out["r"] = float(np.corrcoef(est, ref)[0, 1]) if len(est) > 2 and est.std() > 0 and ref.std() > 0 else float("nan")
    return out


def attach_reference(rows: list[dict], gt: GroundTruth, window: float, prefer: str) -> None:
    for r in rows:
        r["ref_bpm"] = gt.reference_bpm(r["t_end"], window, prefer)


def summarize(name: str, rows: list[dict], include_low_signal: bool = True) -> dict:
    ok_status = {"ok", "low_signal"} if include_low_signal else {"ok"}
    used = [r for r in rows if r["status"] in ok_status and r.get("ref_bpm") and r["est_bpm"]]
    coverage = 100 * len(used) / max(1, len(rows))
    m = metrics(np.array([r["est_bpm"] for r in used]), np.array([r["ref_bpm"] for r in used]))
    m.update(name=name, windows=len(rows), coverage=coverage,
             rejected_motion=sum(r["status"] == "motion" for r in rows),
             rejected_noface=sum(r["status"] == "no_face" for r in rows))
    return m


def print_table(summaries: list[dict]) -> None:
    cols = ["name", "windows", "coverage", "MAE", "RMSE", "MAPE%", "r", "pct<=5bpm", "bias", "LoA95", "rejected_motion"]
    print("\n" + "  ".join(f"{c:>15}" for c in cols))
    for s in summaries:
        cells = []
        for c in cols:
            v = s.get(c, "")
            cells.append(f"{v:>15.2f}" if isinstance(v, float) else f"{str(v):>15}")
        print("  ".join(cells))


def plot(rows: list[dict], path: str, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    used = [r for r in rows if r["est_bpm"] and r.get("ref_bpm") and r["status"] in ("ok", "low_signal")]
    if not used:
        print("Nothing to plot.")
        return
    est = np.array([r["est_bpm"] for r in used])
    ref = np.array([r["ref_bpm"] for r in used])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    lo, hi = min(est.min(), ref.min()) - 5, max(est.max(), ref.max()) + 5
    ax[0].scatter(ref, est, s=10, alpha=0.6)
    ax[0].plot([lo, hi], [lo, hi], "k--", lw=1)
    ax[0].set(xlabel="Reference HR (BPM)", ylabel="rPPG HR (BPM)", title="Agreement", xlim=(lo, hi), ylim=(lo, hi))
    mean, diff = (est + ref) / 2, est - ref
    b, s = diff.mean(), 1.96 * diff.std()
    ax[1].scatter(mean, diff, s=10, alpha=0.6)
    for y, ls in ((b, "-"), (b + s, "--"), (b - s, "--")):
        ax[1].axhline(y, color="k", ls=ls, lw=1)
    ax[1].set(xlabel="Mean of rPPG and reference (BPM)", ylabel="rPPG - reference (BPM)",
              title=f"Bland-Altman (bias {b:+.1f}, LoA ±{s:.1f})")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"Saved plot -> {path}")


def write_csv(rows: list[dict], path: str) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"Saved per-window results -> {path}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Validate rPPG heart rate against ground truth.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_syn = sub.add_parser("synthetic", help="built-in synthetic face video with known HR")
    p_syn.add_argument("--duration", type=float, default=60)
    p_syn.add_argument("--hr", type=float, default=72)
    p_vid = sub.add_parser("video", help="any video file, optional ground truth")
    p_vid.add_argument("path")
    p_vid.add_argument("--gt", help="ground_truth.txt / gtdump.xmp (UBFC), .json (PURE) or .csv with t,bpm")
    p_ubfc = sub.add_parser("ubfc", help="every subject under a UBFC-rPPG folder")
    p_ubfc.add_argument("root")
    p_pure = sub.add_parser("pure", help="every sequence under a PURE folder")
    p_pure.add_argument("root")

    for p in (p_syn, p_vid, p_ubfc, p_pure):
        p.add_argument("--method", default="POS", choices=["POS", "CHROM", "GREEN", "all"])
        p.add_argument("--window", type=float, default=10.0, help="HR window in seconds")
        p.add_argument("--step", type=float, default=1.0)
        p.add_argument("--motion-threshold", type=float, default=1.0)
        p.add_argument("--ref", default="ppg", choices=["ppg", "hr"],
                       help="reference from contact-PPG spectrum (default) or device HR readings")
        p.add_argument("--max-seconds", type=float)
        p.add_argument("--limit", type=int, help="only the first N subjects (ubfc/pure)")
        p.add_argument("--out", help="per-window CSV output")
        p.add_argument("--plot", help="save scatter + Bland-Altman PNG")
    a = ap.parse_args(argv)

    base = EstimatorConfig(hr_window=a.window, motion_threshold=a.motion_threshold)
    methods = ["POS", "CHROM", "GREEN"] if a.method == "all" else [a.method]

    # 1. Build list of (name, frame-iterator factory, ground truth)
    jobs = []
    if a.cmd == "synthetic":
        spec = SyntheticSpec(duration=a.duration, hr_bpm=a.hr)
        t = np.arange(0, spec.duration, 1 / 60)
        gt = GroundTruth(hr_t=t, hr=spec.true_hr(t))
        jobs.append(("synthetic", lambda: synthetic_frames(spec), gt))
        a.ref = "hr"
    elif a.cmd == "video":
        gt = load_auto(a.gt) if a.gt else None
        jobs.append((Path(a.path).name, lambda: video_frames(a.path), gt))
    elif a.cmd == "ubfc":
        for name, vid, gtp in find_ubfc_subjects(a.root):
            jobs.append((name, lambda v=vid: video_frames(v), load_ubfc(gtp)))
    elif a.cmd == "pure":
        for name, frames, js in find_pure_sequences(a.root):
            jobs.append((name, lambda f=frames: pure_frames(f), load_pure(js)))
    if a.limit:
        jobs = jobs[: a.limit]
    if not jobs:
        sys.exit("No videos found.")

    # 2. Track faces once per video, then run every method on the traces.
    all_rows: dict[str, list[dict]] = {m: [] for m in methods}
    per_subject: dict[str, list[dict]] = {m: [] for m in methods}
    with FaceROITracker() as tracker:
        for name, frames, gt in jobs:
            tr = extract_traces(frames(), tracker, a.max_seconds, label=name)
            for m in methods:
                rows = sliding_hr(tr, replace(base, method=m), a.step)
                for r in rows:
                    r["subject"] = name
                if gt is not None:
                    attach_reference(rows, gt, a.window, a.ref)
                    per_subject[m].append(summarize(name, rows))
                all_rows[m].extend(rows)

    # 3. Report
    has_gt = any(gt is not None for *_, gt in jobs)
    for m in methods:
        rows = all_rows[m]
        suffix = f"_{m}" if len(methods) > 1 else ""
        print(f"\n=== {m} ===")
        if a.out:
            write_csv(rows, a.out.replace(".csv", f"{suffix}.csv"))
        if not has_gt:
            for r in rows:
                est = f"{r['est_bpm']:.1f}" if r["est_bpm"] else "-"
                print(f"t={r['t_end']:7.1f}s  {r['status']:<11} HR={est}")
            continue
        summaries = per_subject[m]
        if len(summaries) > 1:
            summaries = summaries + [summarize("ALL", rows)]
        print_table(summaries)
        if a.plot:
            plot(rows, a.plot.replace(".png", f"{suffix}.png"), f"{m} - {a.cmd}")
    if has_gt:
        print("\nMAE/RMSE in BPM; coverage = % of windows not rejected; r = Pearson; "
              "LoA95 = Bland-Altman limits of agreement (±).")


if __name__ == "__main__":
    main()
