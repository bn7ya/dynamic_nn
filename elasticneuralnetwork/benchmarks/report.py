from __future__ import annotations

import argparse
import csv
from pathlib import Path

from elasticneuralnetwork.benchmarks._common import RESULTS_DIR


def aggregate_results(results_dir: Path = RESULTS_DIR):
    rows = []
    for csv_path in sorted(results_dir.glob("*.csv")):
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            rows.extend(list(reader))
    return rows


def write_markdown_table(rows, dest: Path):
    if not rows:
        dest.write_text("(no results)\n")
        return
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(h, "")) for h in headers)
                      + " |")
    dest.write_text("\n".join(lines) + "\n")


def scatter_params_vs_accuracy(rows, dest: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    methods = sorted({r["method"] for r in rows})
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in methods:
        sub = [r for r in rows if r["method"] == m]
        xs = [float(r["parameter_count"]) for r in sub]
        ys = [float(r["accuracy"]) for r in sub]
        ax.scatter(xs, ys, label=m, alpha=0.7)
    ax.set_xlabel("parameter count")
    ax.set_ylabel("accuracy")
    ax.set_xscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(dest, dpi=100)
    plt.close(fig)


def verification_criterion_4(rows) -> dict:
    by_dataset = {}
    for r in rows:
        by_dataset.setdefault(r["dataset"], {})[r["method"]] = r
    wins = 0
    detail = {}
    for dataset, by_method in by_dataset.items():
        enn = by_method.get("elasticneuralnetwork")
        baseline = (by_method.get("pytorch_cnn")
                    or by_method.get("pytorch_mlp"))
        if not enn or not baseline:
            continue
        enn_acc = float(enn["accuracy"])
        bl_acc = float(baseline["accuracy"])
        enn_params = int(enn["parameter_count"])
        bl_params = int(baseline["parameter_count"])
        param_reduction = (1.0 - enn_params / max(1, bl_params))
        beat = enn_acc >= bl_acc and param_reduction >= 0.20
        detail[dataset] = {
            "enn_accuracy": enn_acc, "baseline_accuracy": bl_acc,
            "enn_params": enn_params, "baseline_params": bl_params,
            "param_reduction": param_reduction, "beat": beat,
        }
        if beat:
            wins += 1
    return {"wins": wins, "needed": 2, "passes": wins >= 2,
             "detail": detail}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--out-md", type=Path,
                    default=RESULTS_DIR / "summary.md")
    p.add_argument("--out-png", type=Path,
                    default=RESULTS_DIR / "scatter.png")
    args = p.parse_args()
    rows = aggregate_results(args.results_dir)
    write_markdown_table(rows, args.out_md)
    if rows:
        scatter_params_vs_accuracy(rows, args.out_png)
    verdict = verification_criterion_4(rows)
    print(f"summary: {args.out_md}")
    print(f"scatter: {args.out_png}")
    print(f"verification criterion 4 (≥2 datasets beat with "
            f"≥20% param reduction): {verdict['wins']} wins; "
            f"passes={verdict['passes']}")
    for dataset, d in verdict["detail"].items():
        marker = "PASS" if d["beat"] else "----"
        print(f"  [{marker}] {dataset}: "
                f"enn acc={d['enn_accuracy']:.4f} "
                f"({d['enn_params']} params) vs baseline "
                f"acc={d['baseline_accuracy']:.4f} "
                f"({d['baseline_params']} params); "
                f"param_reduction={d['param_reduction']:.2%}")


if __name__ == "__main__":
    main()
