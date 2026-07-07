#!/usr/bin/env python3
"""Aggregate per-case loco baseline summaries into a final report.

Walks a baseline directory, collects all summary_loco.json files,
computes pass/fail per test, and produces:
  - loco_baseline_report.json  (machine-readable)
  - loco_baseline_report.md    (human-readable)

Supports model comparison with --compare-dir.

Usage:
    python loco_report.py --baseline-dir video/loco_baseline/20260614_120000_default \\
        --config loco_config.json [--compare-dir <other_dir>]
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path


def find_summaries(baseline_dir):
    """Walk baseline_dir and collect all summary_loco.json paths.

    Returns dict: {(test_id, case_id, run_label): path}
    """
    summaries = {}
    root = Path(baseline_dir)
    if not root.is_dir():
        return summaries
    for summary_path in sorted(root.rglob("summary_loco.json")):
        rel = summary_path.relative_to(root)
        parts = rel.parts
        if len(parts) >= 3:
            test_id = parts[0]
            case_id = parts[1]
            run_label = parts[2]
            summaries[(test_id, case_id, run_label)] = str(summary_path)
        elif len(parts) == 2:
            test_id = parts[0]
            case_id = parts[1]
            run_label = "run_001"
            summaries[(test_id, case_id, run_label)] = str(summary_path)
    return summaries


def load_summary(path):
    with open(path) as fh:
        return json.load(fh)


def aggregate(baseline_dir, config_path=None):
    """Aggregate all summaries into a structured report dict."""
    summaries = find_summaries(baseline_dir)
    if not summaries:
        return {"error": "No summary_loco.json files found", "dir": str(baseline_dir)}

    # Group by test_id -> case_id -> list of runs
    tree = defaultdict(lambda: defaultdict(list))
    for (test_id, case_id, run_label), path in summaries.items():
        s = load_summary(path)
        s["_run_label"] = run_label
        tree[test_id][case_id].append(s)

    # Load criteria from config
    criteria_map = {}
    acceptance = {}
    if config_path and os.path.exists(config_path):
        with open(config_path) as fh:
            cfg = json.load(fh)
        for tid, tcfg in cfg.get("tests", {}).items():
            criteria_map[tid] = tcfg.get("criteria", {})
        acceptance = cfg.get("acceptance_rating", {})

    per_test = {}
    totals = {"pass": 0, "fail": 0, "info": 0}

    for test_id in sorted(tree.keys()):
        test_entry = {"test_name": test_id, "cases": {}, "overall_pass": None}
        all_pass = True
        any_result = False

        for case_id in sorted(tree[test_id].keys()):
            runs = tree[test_id][case_id]
            case_pass = True
            run_entries = []

            for s in runs:
                run_label = s.pop("_run_label", "?")
                p = s.get("pass", {})
                overall = p.get("overall", None)
                run_entries.append({
                    "run": run_label,
                    "pass": overall,
                    "criteria": p,
                    "fall_frames": s.get("fall_frames", 0),
                    "average_hz": s.get("average_hz", 0),
                    "vx_error_percent": s.get("steady_velocity", {}).get("vx_error_percent"),
                    "displacement": s.get("displacement", {}),
                })
                any_result = True
                if overall is False:
                    case_pass = False
                    all_pass = False

            test_entry["cases"][case_id] = {
                "runs": run_entries,
                "case_pass": case_pass,
            }

            if case_pass:
                totals["pass"] += 1
            else:
                totals["fail"] += 1

        test_entry["overall_pass"] = all_pass if any_result else None
        per_test[test_id] = test_entry

    # Compute acceptance rating
    rating = compute_rating(per_test, acceptance)

    return {
        "meta": {
            "baseline_dir": str(baseline_dir),
            "total_cases_with_results": totals["pass"] + totals["fail"],
            "pass_count": totals["pass"],
            "fail_count": totals["fail"],
            "info_count": totals["info"],
        },
        "per_test": per_test,
        "acceptance_rating": rating,
    }


def compute_rating(per_test, acceptance_cfg):
    """Determine acceptance rating from per-test results."""
    if not acceptance_cfg:
        return {"label": "unknown", "description": "No acceptance config available"}

    t0_to_t8 = ["T0_static_stability", "T1_forward_velocity", "T2_backward",
                 "T3_lateral", "T4_turning", "T5_combined_arcs",
                 "T6_step_response", "T7_endurance", "T8_ball_push"]

    t0_t7 = t0_to_t8[:-1]  # T0-T7

    t0_t8_all_pass = all(
        per_test.get(tid, {}).get("overall_pass") is True
        for tid in t0_to_t8
    )
    t0_t7_all_pass = all(
        per_test.get(tid, {}).get("overall_pass") is True
        for tid in t0_t7
    )

    t0_pass = per_test.get("T0_static_stability", {}).get("overall_pass")
    t1_pass = per_test.get("T1_forward_velocity", {}).get("overall_pass")
    t4_pass = per_test.get("T4_turning", {}).get("overall_pass")

    any_falls = False
    for tid in t0_to_t8:
        for case_id, case_data in per_test.get(tid, {}).get("cases", {}).items():
            for run in case_data.get("runs", []):
                if run.get("fall_frames", 0) > 0:
                    any_falls = True

    if t0_t8_all_pass and not any_falls:
        rating = acceptance_cfg.get("match_ready", {})
    elif t0_t7_all_pass:
        rating = acceptance_cfg.get("marginal", {})
    elif t0_pass is False or t1_pass is False or t4_pass is False:
        rating = acceptance_cfg.get("not_recommended", {})
    elif any_falls:
        rating = acceptance_cfg.get("not_recommended", {})
    else:
        rating = acceptance_cfg.get("marginal", {})

    return rating


def generate_markdown(report, output_path, config_path=None):
    """Write human-readable markdown report."""
    meta = report.get("meta", {})
    per_test = report.get("per_test", {})
    rating = report.get("acceptance_rating", {})

    lines = []
    lines.append("# Locomotion Baseline Report")
    lines.append("")
    lines.append(f"**Baseline:** `{meta.get('baseline_dir', '?')}`")
    lines.append(f"**Cases with results:** {meta.get('total_cases_with_results', 0)}")
    lines.append(f"**Pass / Fail:** {meta.get('pass_count', 0)} / {meta.get('fail_count', 0)}")
    lines.append("")
    lines.append(f"## Acceptance Rating: **{rating.get('label', '?')}**")
    lines.append("")
    if rating.get("description"):
        lines.append(f"> {rating['description']}")
    lines.append("")

    lines.append("## Summary Table")
    lines.append("")
    lines.append("| Test | Case | Runs | Result | Key Metrics |")
    lines.append("|------|------|------|--------|-------------|")

    for test_id in sorted(per_test.keys()):
        entry = per_test[test_id]
        for case_id in sorted(entry.get("cases", {}).keys()):
            case = entry["cases"][case_id]
            runs = case.get("runs", [])
            n_runs = len(runs)
            case_pass = case.get("case_pass", False)
            result = ":white_check_mark: PASS" if case_pass else ":x: FAIL"

            # Build key metrics string
            metrics_parts = []
            for r in runs:
                ff = r.get("fall_frames", 0)
                if ff > 0:
                    metrics_parts.append(f"falls={ff}")
            if runs:
                r0 = runs[0]
                vx_err = r0.get("vx_error_percent")
                if vx_err is not None:
                    metrics_parts.append(f"vx_err={vx_err:.1f}%")
                hz = r0.get("average_hz", 0)
                if hz > 0:
                    metrics_parts.append(f"hz={hz:.0f}")
            metrics_str = ", ".join(metrics_parts) if metrics_parts else "-"

            lines.append(f"| {test_id} | {case_id} | {n_runs} | {result} | {metrics_str} |")

    lines.append("")

    # Detailed per-test section
    lines.append("## Details")
    lines.append("")
    for test_id in sorted(per_test.keys()):
        entry = per_test[test_id]
        overall = entry.get("overall_pass")
        status = "PASS" if overall else "FAIL" if overall is False else "INFO"
        lines.append(f"### {test_id} — {status}")
        lines.append("")
        for case_id in sorted(entry.get("cases", {}).keys()):
            case = entry["cases"][case_id]
            for r in case.get("runs", []):
                ff = r.get("fall_frames", 0)
                disp = r.get("displacement", {})
                lines.append(f"- **{case_id}** / {r['run']}: "
                           f"vx_err={r.get('vx_error_percent', '?')}%, "
                           f"falls={ff}, "
                           f"drift={disp.get('max_displacement_m', '?')}m, "
                           f"hz={r.get('average_hz', '?')}")
        lines.append("")

    with open(output_path, "w") as fh:
        fh.write("\n".join(lines))
    return output_path


def generate_json(report, output_path):
    with open(output_path, "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    return output_path


def compare_reports(report_a, report_b):
    """Side-by-side comparison of two baseline reports."""
    per_a = report_a.get("per_test", {})
    per_b = report_b.get("per_test", {})

    comparison = {"test_comparisons": {}}
    all_test_ids = sorted(set(per_a.keys()) | set(per_b.keys()))

    for tid in all_test_ids:
        entry_a = per_a.get(tid, {})
        entry_b = per_b.get(tid, {})
        comparison["test_comparisons"][tid] = {
            "a_overall_pass": entry_a.get("overall_pass"),
            "b_overall_pass": entry_b.get("overall_pass"),
            "a_cases": {},
            "b_cases": {},
        }

    comparison["rating"] = {
        "a": report_a.get("acceptance_rating", {}).get("label", "?"),
        "b": report_b.get("acceptance_rating", {}).get("label", "?"),
    }
    return comparison


def main():
    parser = argparse.ArgumentParser(description="Aggregate loco baseline results")
    parser.add_argument("--baseline-dir", required=True, help="Baseline output directory")
    parser.add_argument("--compare-dir", default=None, help="Second baseline directory for comparison")
    parser.add_argument("--config", default=None, help="Path to loco_config.json")
    parser.add_argument("--output-md", default=None, help="Output markdown path")
    parser.add_argument("--output-json", default=None, help="Output JSON path")
    args = parser.parse_args()

    config_path = args.config or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "loco_config.json"
    )

    report = aggregate(args.baseline_dir, config_path)

    if "error" in report:
        print(f"[ERROR] {report['error']}", file=sys.stderr)
        sys.exit(1)

    # Write outputs
    md_path = args.output_md or os.path.join(args.baseline_dir, "loco_baseline_report.md")
    json_path = args.output_json or os.path.join(args.baseline_dir, "loco_baseline_report.json")

    generate_markdown(report, md_path)
    print(f"Markdown report: {md_path}")

    generate_json(report, json_path)
    print(f"JSON report:    {json_path}")

    # Comparison mode
    if args.compare_dir:
        compare_report = aggregate(args.compare_dir, config_path)
        if "error" not in compare_report:
            comp = compare_reports(report, compare_report)
            comp_path = os.path.join(
                os.path.dirname(md_path),
                "loco_comparison.json"
            )
            generate_json(comp, comp_path)
            print(f"Comparison:     {comp_path}")

    # Print acceptance rating
    rating = report.get("acceptance_rating", {})
    print(f"\nAcceptance: {rating.get('label', '?')}")
    if rating.get("description"):
        print(f"  {rating['description']}")


if __name__ == "__main__":
    main()
