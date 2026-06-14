"""Compare trajectory CSVs for guard-active repeatability validation.

Read-only — no imports outside stdlib.  Takes one or more trajectory CSV
paths and outputs per-run metrics plus a side-by-side comparison table.
"""

import csv
import sys
from collections import Counter
from pathlib import Path


def _f(v):
    """Format a float or None."""
    if v is None:
        return "N/A"
    return f"{float(v):.3f}"


def analyze(path, label=None):
    if label is None:
        label = Path(path).parent.name

    with open(path) as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    rows60 = [r for r in rows if r.get("elapsed_s") and float(r["elapsed_s"]) <= 60.0]
    has_guard_cols = all(
        c in (rows60[0] if rows60 else {})
        for c in ["approach_guard_input_vx", "approach_guard_applied"]
    )

    # -- Milestones -------------------------------------------------------
    first_la = first_ap = first_cand = None
    for r in rows60:
        t = float(r.get("elapsed_s", 0))
        if first_la is None and r.get("align_mode") == "LATERAL_ALIGN":
            first_la = (int(r["frame"]), t)
        if first_ap is None and r.get("align_mode") == "ALIGN_PUSH":
            first_ap = (int(r["frame"]), t)
        if first_cand is None and r.get("can_kick_candidate") in ("true", "True", "1"):
            first_cand = (int(r["frame"]), t)

    # -- Mode distribution ------------------------------------------------
    modes = Counter(r.get("align_mode", "") for r in rows60)

    # -- Guard rate -------------------------------------------------------
    app = [r for r in rows60 if r.get("align_mode") == "APPROACH"]
    gt = [r for r in app if r.get("approach_guard_applied") in ("True", "true", "1")]
    guard_pct = 100 * len(gt) / max(1, len(app)) if has_guard_cols else None

    # -- Counts -----------------------------------------------------------
    cand_count = sum(1 for r in rows60 if r.get("can_kick_candidate") in ("true", "True", "1"))
    lat_err_count = sum(1 for r in rows60 if r.get("can_kick_reason") == "lateral_error")
    fallen_rows = [r for r in rows60 if r.get("is_fallen") in ("True", "true", "1")]
    fallen_count = len(fallen_rows)
    first_fallen = float(fallen_rows[0]["elapsed_s"]) if fallen_rows else None

    # -- APPROACH quality -------------------------------------------------
    app_lat = [abs(float(r["lateral_err"])) for r in app if r.get("lateral_err")]
    mean_abs_lat = sum(app_lat) / len(app_lat) if app_lat else None

    # -- Pre-ALIGN_PUSH 5s buckets ----------------------------------------
    ap_cutoff = first_ap[1] if first_ap else 61.0
    pre_ap = [r for r in rows60 if float(r.get("elapsed_s", 0)) < ap_cutoff]
    buckets = {}
    for bstart in range(0, int(ap_cutoff) + 5, 5):
        bend = min(bstart + 5, ap_cutoff)
        bro = [r for r in pre_ap if bstart <= float(r.get("elapsed_s", 0)) < bend]
        if not bro:
            continue
        bd = [float(r["ball_distance"]) for r in bro if r.get("ball_distance")]
        bl = [abs(float(r["lateral_err"])) for r in bro if r.get("lateral_err")]
        bm = Counter(r.get("align_mode", "") for r in bro)
        vx = [float(r["cmd_vx"]) for r in bro if r.get("cmd_vx")]
        vy = [float(r["cmd_vy"]) for r in bro if r.get("cmd_vy")]
        w = [float(r["cmd_w"]) for r in bro if r.get("cmd_w")]
        buckets[bstart] = {
            "n": len(bro),
            "ball_dist_mean": sum(bd) / len(bd) if bd else None,
            "lat_mean": sum(bl) / len(bl) if bl else None,
            "modes": dict(bm),
            "vx_mean": sum(vx) / len(vx) if vx else None,
            "vy_mean": sum(vy) / len(vy) if vy else None,
            "w_mean": sum(w) / len(w) if w else None,
        }

    return {
        "label": label,
        "total_frames": len(rows),
        "frames_60s": len(rows60),
        "has_guard_cols": has_guard_cols,
        "guard_pct": guard_pct,
        "first_la": first_la,
        "first_ap": first_ap,
        "first_cand": first_cand,
        "modes": dict(modes),
        "cand_count": cand_count,
        "lat_err_count": lat_err_count,
        "fallen_count": fallen_count,
        "first_fallen": first_fallen,
        "app_frames": len(app),
        "mean_abs_lat_app": mean_abs_lat,
        "buckets": buckets,
    }


def print_run(run):
    print(f"--- {run['label']} ---")
    print(f"  Frames: {run['total_frames']} total, {run['frames_60s']} in first 60s")
    print(f"  Guard columns: {run['has_guard_cols']}")
    if run["guard_pct"] is not None:
        print(f"  Guard true rate: {run['guard_pct']:.0f}% ({run['app_frames']} APPROACH frames)")
    for name, val in [("LATERAL_ALIGN", run["first_la"]),
                      ("ALIGN_PUSH", run["first_ap"]),
                      ("candidate", run["first_cand"])]:
        if val:
            print(f"  First {name}: frame={val[0]} elapsed_s={val[1]:.1f}")
        else:
            print(f"  First {name}: NEVER in 60s")
    print(f"  candidate_count: {run['cand_count']}")
    print(f"  lateral_error_count: {run['lat_err_count']}")
    print(f"  fallen_count: {run['fallen_count']}" +
          (f"  first_fallen: {run['first_fallen']:.1f}s" if run["first_fallen"] else ""))
    if run["mean_abs_lat_app"] is not None:
        print(f"  APPROACH mean abs(lateral_err): {run['mean_abs_lat_app']:.4f}")
    print(f"  Mode distribution (60s): {run['modes']}")

    if run["buckets"]:
        ap_s = run["first_ap"][1] if run["first_ap"] else 61.0
        print(f"  Pre-ALIGN_PUSH 5s buckets (0-{ap_s:.1f}s):")
        for bstart in sorted(run["buckets"]):
            b = run["buckets"][bstart]
            bend = min(bstart + 5, ap_s)
            print(f"    [{bstart:2d}-{int(bend):2d}s] n={b['n']:3d}  "
                  f"ball_dist={_f(b['ball_dist_mean'])}  "
                  f"|lat|={_f(b['lat_mean'])}  "
                  f"vx={_f(b['vx_mean'])} vy={_f(b['vy_mean'])} w={_f(b['w_mean'])}  "
                  f"modes={b['modes']}")
    print()


def print_summary(runs):
    print("=== Summary Table ===")
    header = f"{'Run':<35} {'guard%':>6} {'cand':>5} {'lat_err':>7} {'falls':>5} {'1stAP':>7} {'|lat|APP':>8}"
    print(header)
    print("-" * len(header))
    for r in runs:
        first_ap_s = f"{r['first_ap'][1]:.1f}s" if r["first_ap"] else "N/A"
        guard_s = f"{r['guard_pct']:.0f}%" if r["guard_pct"] is not None else "N/A"
        lat_s = f"{r['mean_abs_lat_app']:.4f}" if r["mean_abs_lat_app"] is not None else "N/A"
        print(f"{r['label']:<35} {guard_s:>6} {r['cand_count']:>5} "
              f"{r['lat_err_count']:>7} {r['fallen_count']:>5} {first_ap_s:>7} {lat_s:>8}")
    print()

    # Variance check
    cands = [r["cand_count"] for r in runs]
    aps = [r["first_ap"][1] for r in runs if r["first_ap"]]
    if len(cands) >= 2:
        print(f"candidate_count range: {min(cands)}-{max(cands)}")
    if len(aps) >= 2:
        print(f"First ALIGN_PUSH range: {min(aps):.1f}s-{max(aps):.1f}s")
    falls = [r["fallen_count"] for r in runs]
    if any(f > 0 for f in falls):
        print("WARNING: falls detected in at least one run")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <trajectory.csv> [trajectory2.csv ...]")
        sys.exit(1)

    runs = []
    for path in sys.argv[1:]:
        label = Path(path).parent.name.replace("trajectory_20260613_", "")
        try:
            runs.append(analyze(path, label))
        except Exception as e:
            print(f"ERROR reading {path}: {e}", file=sys.stderr)
            raise

    for run in runs:
        print_run(run)
    print_summary(runs)


if __name__ == "__main__":
    main()
