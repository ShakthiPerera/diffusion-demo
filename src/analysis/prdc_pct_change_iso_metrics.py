"""
PRDC percentage-change analysis across ISO metric variants relative to baseline.

Workflow requested:
  1. Scan every PRDC checkpoint CSV (``checkpoint_prdc_metrics_*.csv``) for the
     baseline objective and each ISO metric variant (trace, bures, frob,
     logeig, split).
  2. For the baseline directory, choose the checkpoint row that maximises the
     PRDC density within the optional step window (preferring regs closest to
     the requested baseline reg, default 0.0).
  3. For the "trace" ISO variant, within an optional inclusive step range,
     retain a checkpoint row whose precision and density both exceed baseline.
     If multiple rows satisfy the condition, keep the one with highest density,
     breaking ties by precision then step.
  4. For other ISO variants (bures, frob, logeig, split), pick the checkpoint
     row with the highest PRDC density (ties resolved by precision then step)
     inside the same step window.
  5. Plot the percentage change (precision, recall, density, coverage) versus
     the baseline, keeping the x-axis order: trace, bures, frob, logeig, split.

Outputs (per dataset):
  results/prdc_pct_change_iso_metrics/<dataset>/<range_label>/
      summary_raw.csv          # selected raw PRDC values (reg, step, metrics)
      summary_pct_change.csv   # % change vs baseline
      prdc_pct_change_vs_iso_metric_<metric><suffix>.png
      selection_log.txt        # textual summary of chosen checkpoints

Example:
  python prdc_pct_change_iso_metrics.py --base_dir outputs \
      --datasets moon_circles --min_step 20000 --max_step 25000
"""

from __future__ import annotations

from pathlib import Path
import argparse
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PRDC_METRICS = ["precision", "recall", "density", "coverage"]
DEFAULT_BASELINE_DIR = "iso_iso_compare_baseline"
DEFAULT_VARIANT_MAP = {
    "trace": "iso_iso_compare",
    "bures": "iso_bures_iso_compare",
    "frob": "iso_frob_iso_compare",
    "logeig": "iso_logeig_iso_compare",
    "split": "iso_split_iso_compare",
}
DEFAULT_VARIANT_ORDER = ["trace", "bures", "frob", "logeig", "split"]


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def parse_reg_dir_name(name: str) -> Optional[float]:
    if not name.startswith("reg_"):
        return None
    try:
        return float(name.split("reg_")[1])
    except Exception:
        return None


def format_reg_value(reg: float) -> str:
    text = f"{reg:.10g}"
    if "e" in text or "E" in text:
        text = f"{reg:.10f}".rstrip("0").rstrip(".")
    if "." not in text:
        text = f"{text}.0"
    return text


def read_prdc_checkpoint(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame(columns=["step", *PRDC_METRICS])
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return pd.DataFrame(columns=["step", *PRDC_METRICS])
    needed = ["step", *PRDC_METRICS]
    if not all(c in df.columns for c in needed) or df.empty:
        return pd.DataFrame(columns=["step", *PRDC_METRICS])
    df = df.sort_values("step").reset_index(drop=True)
    return df[needed]


def ensure_outdir(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


# ---------------------------------------------------------------------------
# Selection helpers
# ---------------------------------------------------------------------------

def choose_baseline_entry(
    baseline_dir: Path,
    dataset: str,
    preferred_reg: Optional[float],
    min_step: Optional[int],
    max_step: Optional[int],
) -> Optional[Dict[str, float]]:
    entries: List[Dict[str, float]] = []
    for reg_dir in sorted(baseline_dir.iterdir()):
        if not reg_dir.is_dir():
            continue
        reg_val = parse_reg_dir_name(reg_dir.name)
        if reg_val is None:
            continue
        reg_str = format_reg_value(reg_val)
        csv_path = reg_dir / "checkpoints" / f"checkpoint_prdc_metrics_{dataset}_reg_{reg_str}.csv"
        df = read_prdc_checkpoint(csv_path)
        if df.empty:
            continue
        if (min_step is not None) or (max_step is not None):
            lo = min_step if min_step is not None else int(df["step"].min())
            hi = max_step if max_step is not None else int(df["step"].max())
            df = df[(df["step"] >= lo) & (df["step"] <= hi)]
            if df.empty:
                continue
        idx = df["density"].idxmax()
        row = df.loc[idx]
        entry = {
            "iso_metric": "baseline",
            "reg": float(reg_val),
            "step": int(row["step"]),
            **{metric: float(row[metric]) for metric in PRDC_METRICS},
        }
        entries.append(entry)
    if not entries:
        return None
    if preferred_reg is not None:
        chosen = min(entries, key=lambda e: abs(e["reg"] - preferred_reg))
    else:
        chosen = min(entries, key=lambda e: (abs(e["reg"]), e["reg"]))
    return chosen.copy()


def select_variant_entry_strict(
    iso_name: str,
    variant_dir: Path,
    dataset: str,
    baseline_metrics: Dict[str, float],
    min_step: Optional[int],
    max_step: Optional[int],
) -> Optional[Dict[str, float]]:
    best_entry: Optional[Dict[str, float]] = None
    for reg_dir in sorted(variant_dir.iterdir()):
        if not reg_dir.is_dir():
            continue
        reg_val = parse_reg_dir_name(reg_dir.name)
        if reg_val is None:
            continue
        reg_str = format_reg_value(reg_val)
        csv_path = reg_dir / "checkpoints" / f"checkpoint_prdc_metrics_{dataset}_reg_{reg_str}.csv"
        df = read_prdc_checkpoint(csv_path)
        if df.empty:
            continue
        if (min_step is not None) or (max_step is not None):
            lo = min_step if min_step is not None else int(df["step"].min())
            hi = max_step if max_step is not None else int(df["step"].max())
            df = df[(df["step"] >= lo) & (df["step"] <= hi)]
            if df.empty:
                continue
        mask = (df["precision"] > baseline_metrics["precision"]) & (df["density"] > baseline_metrics["density"])
        df_valid = df[mask]
        if df_valid.empty:
            continue
        df_valid = df_valid.sort_values(by=["density", "precision", "step"], ascending=[False, False, False])
        row = df_valid.iloc[0]
        entry = {
            "iso_metric": iso_name,
            "reg": float(reg_val),
            "step": int(row["step"]),
        }
        entry.update({metric: float(row[metric]) for metric in PRDC_METRICS})
        if (
            best_entry is None
            or entry["density"] > best_entry["density"]
            or (
                entry["density"] == best_entry["density"]
                and (entry["precision"], entry["step"]) > (best_entry["precision"], best_entry["step"])
            )
        ):
            best_entry = entry
    return best_entry


def select_variant_entry_best_density(
    iso_name: str,
    variant_dir: Path,
    dataset: str,
    min_step: Optional[int],
    max_step: Optional[int],
) -> Optional[Dict[str, float]]:
    best_entry: Optional[Dict[str, float]] = None
    for reg_dir in sorted(variant_dir.iterdir()):
        if not reg_dir.is_dir():
            continue
        reg_val = parse_reg_dir_name(reg_dir.name)
        if reg_val is None:
            continue
        reg_str = format_reg_value(reg_val)
        csv_path = reg_dir / "checkpoints" / f"checkpoint_prdc_metrics_{dataset}_reg_{reg_str}.csv"
        df = read_prdc_checkpoint(csv_path)
        if df.empty:
            continue
        if (min_step is not None) or (max_step is not None):
            lo = min_step if min_step is not None else int(df["step"].min())
            hi = max_step if max_step is not None else int(df["step"].max())
            df = df[(df["step"] >= lo) & (df["step"] <= hi)]
            if df.empty:
                continue
        idx = df["density"].idxmax()
        row = df.loc[idx]
        entry = {
            "iso_metric": iso_name,
            "reg": float(reg_val),
            "step": int(row["step"]),
        }
        entry.update({metric: float(row[metric]) for metric in PRDC_METRICS})
        if (
            best_entry is None
            or entry["density"] > best_entry["density"]
            or (
                entry["density"] == best_entry["density"]
                and (entry["precision"], entry["step"]) > (best_entry["precision"], best_entry["step"])
            )
        ):
            best_entry = entry
    return best_entry


def compute_pct_change(baseline: Dict[str, float], variant: Dict[str, float]) -> Dict[str, float]:
    pct: Dict[str, float] = {}
    for metric in PRDC_METRICS:
        base_val = baseline.get(metric, np.nan)
        var_val = variant.get(metric, np.nan)
        if np.isnan(base_val) or base_val == 0.0 or np.isnan(var_val):
            pct[metric] = np.nan
        else:
            pct[metric] = ((var_val - base_val) / base_val) * 100.0
    return pct


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_pct_change(
    df_pct: pd.DataFrame,
    dataset: str,
    out_dir: Path,
    suffix: str,
    order: List[str],
) -> None:
    labels = [name for name in order if name in df_pct["iso_metric"].values]
    if not labels:
        return
    df_plot = df_pct.set_index("iso_metric").loc[labels]
    x_idx = np.arange(len(labels))
    xticks = [label.replace("_", " ") for label in labels]
    for metric in PRDC_METRICS:
        if metric not in df_plot.columns:
            continue
        values = df_plot[metric].values
        fig = plt.figure()
        ax = fig.add_subplot(111)
        bars = ax.bar(x_idx, values)
        for bar, val in zip(bars, values):
            if np.isnan(val):
                continue
            ax.annotate(f"{val:+.2f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, val),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=8)
        ax.set_xticks(x_idx)
        ax.set_xticklabels(xticks, rotation=15, ha="right")
        ax.set_ylabel("% change vs baseline")
        ax.set_xlabel("ISO metric variant")
        ax.set_title(f"{dataset} — {metric.capitalize()} % change")
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)
        fig.tight_layout()
        fig.savefig(out_dir / f"prdc_pct_change_vs_iso_metric_{metric}{suffix}.png", dpi=200)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def process_dataset(
    base_dir: Path,
    dataset: str,
    iso_metrics: List[str],
    variant_map: Dict[str, str],
    baseline_dir_name: str,
    baseline_reg_preference: Optional[float],
    min_step: Optional[int],
    max_step: Optional[int],
) -> None:
    dataset_dir = base_dir / dataset
    if not dataset_dir.exists():
        print(f"[WARN] Dataset not found: {dataset_dir}")
        return

    baseline_dir = dataset_dir / baseline_dir_name
    if not baseline_dir.exists():
        print(f"[WARN] {dataset}: baseline directory '{baseline_dir_name}' missing.")
        return

    baseline_entry = choose_baseline_entry(
        baseline_dir,
        dataset,
        preferred_reg=baseline_reg_preference,
        min_step=min_step,
        max_step=max_step,
    )
    if not baseline_entry:
        print(f"[WARN] {dataset}: could not determine baseline PRDC metrics.")
        return
    baseline_metrics = {metric: baseline_entry[metric] for metric in PRDC_METRICS}

    selected_raw: List[Dict[str, float]] = []
    selected_pct: List[Dict[str, float]] = []
    selection_msgs: List[str] = []

    # Record baseline first
    baseline_label = "baseline"
    baseline_row = {
        "iso_metric": baseline_label,
        "reg": baseline_entry["reg"],
        "step": baseline_entry["step"],
    }
    for metric in PRDC_METRICS:
        baseline_row[metric] = baseline_entry[metric]
    selected_raw.append(baseline_row)
    selected_pct.append({
        "iso_metric": baseline_label,
        "reg": baseline_entry["reg"],
        "step": baseline_entry["step"],
        **{metric: 0.0 for metric in PRDC_METRICS},
    })
    selection_msgs.append(
        f"baseline: reg={baseline_entry['reg']}, step={baseline_entry['step']}"
        f" | metrics={{{', '.join(f'{m}={baseline_entry[m]:.6f}' for m in PRDC_METRICS)}}}"
    )

    for iso_name in iso_metrics:
        variant_dir_name = variant_map.get(iso_name, iso_name)
        variant_dir = dataset_dir / variant_dir_name
        if not variant_dir.exists():
            selection_msgs.append(f"{iso_name}: directory '{variant_dir_name}' missing.")
            continue
        if iso_name == "trace":
            entry = select_variant_entry_strict(
                iso_name=iso_name,
                variant_dir=variant_dir,
                dataset=dataset,
                baseline_metrics=baseline_metrics,
                min_step=min_step,
                max_step=max_step,
            )
        else:
            entry = select_variant_entry_best_density(
                iso_name=iso_name,
                variant_dir=variant_dir,
                dataset=dataset,
                min_step=min_step,
                max_step=max_step,
            )
        if not entry:
            if iso_name == "trace":
                selection_msgs.append(f"{iso_name}: no checkpoint exceeded baseline precision & density.")
            else:
                selection_msgs.append(f"{iso_name}: no checkpoints available after filtering.")
            continue
        row = {
            "iso_metric": iso_name,
            "reg": entry["reg"],
            "step": entry["step"],
        }
        for metric in PRDC_METRICS:
            row[metric] = entry[metric]
        selected_raw.append(row)

        pct_row = {
            "iso_metric": iso_name,
            "reg": entry["reg"],
            "step": entry["step"],
        }
        pct_row.update(compute_pct_change(baseline_metrics, entry))
        selected_pct.append(pct_row)

        selection_msgs.append(
            f"{iso_name}: reg={entry['reg']}, step={entry['step']} "
            f"| metrics={{{', '.join(f'{m}={entry[m]:.6f}' for m in PRDC_METRICS)}}}"
        )

    if len(selected_raw) <= 1:
        print(f"[WARN] {dataset}: no ISO variants satisfied the precision & density criteria.")
        return

    range_label = f"steps_{min_step if min_step is not None else 'min'}-{max_step if max_step is not None else 'max'}"
    out_dir = ensure_outdir(base_dir / "results" / "prdc_pct_change_iso_metrics" / dataset / range_label)

    suffix = ""
    if (min_step is not None) or (max_step is not None):
        suffix = f"_steps-{min_step if min_step is not None else 'min'}-{max_step if max_step is not None else 'max'}"

    df_raw = pd.DataFrame(selected_raw)
    df_pct = pd.DataFrame(selected_pct)

    df_raw.to_csv(out_dir / "summary_raw.csv", index=False)
    df_pct.to_csv(out_dir / "summary_pct_change.csv", index=False)
    (out_dir / "selection_log.txt").write_text("\n".join(selection_msgs) + "\n")

    plot_pct_change(df_pct[df_pct["iso_metric"] != "baseline"], dataset, out_dir, suffix, order=iso_metrics)
    print(f"[OK] {dataset}: analysis written to {out_dir}")


def discover_datasets(base_dir: Path) -> List[str]:
    return sorted([p.name for p in base_dir.iterdir() if p.is_dir() and p.name != "results"])


def main() -> None:
    ap = argparse.ArgumentParser(description="PRDC % change vs baseline across ISO metric variants.")
    ap.add_argument("--base_dir", type=str, default="outputs", help="Root folder containing dataset directories.")
    ap.add_argument("--datasets", type=str, nargs="+", default=None, help="Datasets to process. Defaults to all datasets under base_dir.")
    ap.add_argument("--iso_metrics", type=str, nargs="*", default=None, help="ISO metric variants to evaluate (default trace bures frob logeig split).")
    ap.add_argument("--baseline_dir", type=str, default=DEFAULT_BASELINE_DIR, help="Baseline directory name (relative to dataset dir).")
    ap.add_argument("--baseline_reg", type=float, default=None, help="Preferred baseline regularisation value (fallback: closest to 0.0).")
    ap.add_argument("--min_step", type=int, default=None, help="Inclusive minimum training step when scanning checkpoints.")
    ap.add_argument("--max_step", type=int, default=None, help="Inclusive maximum training step when scanning checkpoints.")
    args = ap.parse_args()

    base = Path(args.base_dir).expanduser().resolve()
    if not base.exists():
        raise FileNotFoundError(f"Base dir not found: {base}")

    if args.datasets:
        datasets = list(args.datasets)
    else:
        datasets = discover_datasets(base)
    if not datasets:
        raise RuntimeError(f"No datasets found in {base}")

    iso_metrics = list(args.iso_metrics) if args.iso_metrics else list(DEFAULT_VARIANT_ORDER)

    for dataset in datasets:
        process_dataset(
            base_dir=base,
            dataset=dataset,
            iso_metrics=iso_metrics,
            variant_map=DEFAULT_VARIANT_MAP,
            baseline_dir_name=args.baseline_dir,
            baseline_reg_preference=args.baseline_reg,
            min_step=args.min_step,
            max_step=args.max_step,
        )


if __name__ == "__main__":
    main()
