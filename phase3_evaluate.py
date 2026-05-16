# ─────────────────────────────────────────────
# phase3_evaluate.py
#
# PHASE 3 — Formal UQ Method Comparison
#
# Runs all 100 test questions through all 4 UQ
# methods and produces:
#   - Comparison table (ECE, AUROC, accuracy, time)
#   - Reliability diagram (all 4 methods overlaid)
#   - ECE bar chart
#   - Per-task breakdown
#   - JSON results for demo use
#
# Usage:
#   cd project9_mas
#   export HF_API_KEY=hf_...
#   PYTHONPATH=. python phase3_evaluate.py
#
# Results saved to: results/phase3/
# ─────────────────────────────────────────────

import os, sys, json, time, math, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Dict
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))

from core.orchestrator import Orchestrator
from core.hf_api       import get_hf_client
import config as cfg

RESULTS_DIR = "results/phase3"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── CLI args (parsed early so main() can use them) ───
_parser = argparse.ArgumentParser(description="Phase 3 — UQ Method Comparison")
_parser.add_argument("--backend", default="auto",
                     choices=["auto", "ollama", "groq", "hf", "mock"],
                     help="Inference backend (default: auto-detect)")
_parser.add_argument("--mock", action="store_true",
                     help="Force mock client (testing)")
_parser.add_argument("--n", type=int, default=None,
                     help="Number of test questions (default: full set)")
_ARGS, _ = _parser.parse_known_args()

UQ_METHODS = ["Softmax", "Temperature Scaling", "MC Dropout", "VIB Layer"]
COLORS     = {"Softmax":"#8892B0", "Temperature Scaling":"#F0A500",
               "MC Dropout":"#A68FD8", "VIB Layer":"#1FC5A8"}


# ── Step 0: Load or generate dataset ─────────

def load_test_set() -> List[Dict]:
    test_path = "data/split_test.json"
    full_path  = "data/actuarial_dataset.json"

    if os.path.exists(test_path):
        with open(test_path) as f:
            data = json.load(f)
        print(f"Loaded {len(data)} test questions from {test_path}")
        return data

    if os.path.exists(full_path):
        with open(full_path) as f:
            all_data = json.load(f)
        import random; random.seed(42); random.shuffle(all_data)
        test = all_data[:100]
        print(f"Using first 100 of {len(all_data)} questions as test set")
        return test

    # Generate fresh
    print("Generating dataset...")
    from data.dataset_generator import generate_dataset, save_dataset
    dataset = generate_dataset(cfg.N_QUESTIONS)
    save_dataset(dataset)
    import random; random.seed(42); random.shuffle(dataset)
    return dataset[:100]


# ── Step 1: Run evaluation ────────────────────

def run_evaluation(
    test_set:  List[Dict],
    api_key:   str,
    n_samples: int = None,   # None = full test set
    verbose:   bool = True,
    backend:   str = None,
    use_mock:  bool = False,
) -> Dict[str, List[Dict]]:
    """
    Run all 4 UQ methods on test_set.
    Returns dict: method_name → list of result dicts.
    """
    if n_samples:
        test_set = test_set[:n_samples]

    print(f"\nRunning evaluation on {len(test_set)} questions × {len(UQ_METHODS)} methods")
    print(f"Total API calls: {len(test_set)} (one per question, scores reused)\n")

    # One orchestrator per UQ method
    all_results = {}

    for method in UQ_METHODS:
        print(f"\n{'─'*55}")
        print(f"  Method: {method}")
        print(f"{'─'*55}")

        orch = Orchestrator(
            api_key    = api_key,
            model_name = cfg.HF_MODEL_SMALL,
            uq_method  = method,
            use_mock   = use_mock,
            backend    = backend,
        )

        results = []
        n_correct = 0

        for i, item in enumerate(test_set):
            t0 = time.time()
            result = orch.run(
                question     = item["question"],
                choices      = item["choices"],
                ground_truth = item.get("answer") or item.get("ground_truth"),
                task         = item.get("task", "unknown"),
                verbose      = False,
            )
            elapsed = time.time() - t0

            # Extract what we need
            record = {
                "id":          item.get("id", i),
                "task":        item.get("task", "unknown"),
                "question":    item["question"][:80],
                "choices":     item["choices"],
                "ground_truth": item.get("answer") or item.get("ground_truth"),
                "predicted":   result.get("predicted"),
                "confidence":  result.get("confidence") or 0,
                "sigma":       result.get("sigma") or 0,
                "is_correct":  result.get("is_correct"),
                "elapsed":     elapsed,
                "uq_method":   method,
                "all_uq":      result.get("all_uq_results") or {},
            }
            results.append(record)

            if record["is_correct"]:
                n_correct += 1

            if verbose and (i + 1) % 10 == 0:
                acc      = n_correct / (i + 1)
                avg_conf = sum(r["confidence"] for r in results) / len(results)
                print(
                    f"  [{i+1:3d}/{len(test_set)}] "
                    f"Acc={acc:.3f}  "
                    f"AvgConf={avg_conf:.3f}  "
                    f"({elapsed:.1f}s last)"
                )

        all_results[method] = results
        final_acc = n_correct / len(results)
        print(f"  → Final accuracy: {final_acc:.4f}")

    return all_results


# ── Step 2: Compute metrics ───────────────────

def compute_metrics(results: List[Dict], method: str) -> Dict:
    """ECE, AUROC, accuracy, mean confidence, mean σ, mean time."""
    confs    = [r["confidence"]  for r in results]
    corrects = [r["is_correct"]  for r in results if r["is_correct"] is not None]
    sigmas   = [r["sigma"]       for r in results]
    times    = [r["elapsed"]     for r in results]

    n        = len(results)
    n_eval   = len(corrects)
    accuracy = sum(corrects) / n_eval if n_eval else 0

    # ECE
    confs_arr = np.array(confs[:n_eval])
    corrs_arr = np.array(corrects, dtype=float)
    bins      = np.linspace(0, 1, cfg.N_BINS + 1)
    ece       = 0.0
    bin_data  = []

    for i in range(cfg.N_BINS):
        lo, hi = bins[i], bins[i+1]
        mask   = (confs_arr >= lo) & (confs_arr < hi)
        if i == cfg.N_BINS - 1:
            mask = (confs_arr >= lo) & (confs_arr <= hi)
        n_bin  = mask.sum()
        if n_bin == 0:
            bin_data.append({"center": (lo+hi)/2, "n": 0, "acc": 0, "conf": 0})
            continue
        acc_b  = float(corrs_arr[mask].mean())
        conf_b = float(confs_arr[mask].mean())
        ece   += (n_bin / n_eval) * abs(acc_b - conf_b)
        bin_data.append({"center": (lo+hi)/2, "n": int(n_bin),
                          "acc": acc_b, "conf": conf_b})

    # AUROC
    auroc = None
    n_pos = sum(corrects)
    n_neg = n_eval - n_pos
    if n_pos > 0 and n_neg > 0:
        order = np.argsort(-confs_arr)
        tp    = np.cumsum(corrs_arr[order])
        fp    = np.cumsum(1 - corrs_arr[order])
        tpr   = tp / n_pos
        fpr   = fp / n_neg
        try:
            auroc = float(np.trapezoid(tpr, fpr))
        except AttributeError:
            auroc = float(np.trapz(tpr, fpr))
        auroc = max(0.0, min(1.0, abs(auroc)))

    # Calibration gap (mean signed)
    cal_gap = float(np.mean(confs_arr) - np.mean(corrs_arr))

    return {
        "method":      method,
        "n":           n,
        "n_eval":      n_eval,
        "accuracy":    float(accuracy),
        "mean_conf":   float(np.mean(confs)),
        "mean_sigma":  float(np.mean(sigmas)),
        "cal_gap":     float(cal_gap),
        "ece":         float(ece),
        "auroc":       float(auroc) if auroc else None,
        "mean_time":   float(np.mean(times)),
        "bin_data":    bin_data,
    }


def per_task_metrics(results: List[Dict]) -> Dict:
    """ECE broken down by task type."""
    tasks = defaultdict(lambda: {"confs": [], "corrects": []})
    for r in results:
        if r["is_correct"] is not None:
            tasks[r["task"]]["confs"].append(r["confidence"])
            tasks[r["task"]]["corrects"].append(r["is_correct"])

    out = {}
    for task, data in tasks.items():
        confs_arr = np.array(data["confs"])
        corrs_arr = np.array(data["corrects"], dtype=float)
        acc       = float(corrs_arr.mean())
        bins      = np.linspace(0, 1, 6)
        ece       = 0.0
        for i in range(5):
            mask  = (confs_arr >= bins[i]) & (confs_arr < bins[i+1])
            n_bin = mask.sum()
            if n_bin == 0: continue
            ece  += (n_bin / len(confs_arr)) * abs(corrs_arr[mask].mean() - confs_arr[mask].mean())
        out[task] = {"ece": float(ece), "accuracy": acc, "n": len(data["confs"])}
    return out


# ── Step 3: Print table ───────────────────────

def print_comparison_table(summaries: List[Dict]):
    cols = ["method", "accuracy", "mean_conf", "cal_gap", "ece", "auroc", "mean_sigma", "mean_time"]
    header = (
        f"{'Method':<22} {'Accuracy':>9} {'MeanConf':>9} "
        f"{'CalGap':>8} {'ECE':>8} {'AUROC':>7} "
        f"{'MeanSigma':>10} {'Time(s)':>8}"
    )
    print(f"\n{'═'*90}")
    print("  PHASE 3 — UQ METHOD COMPARISON")
    print(f"{'═'*90}")
    print(header)
    print(f"{'─'*90}")

    best_ece   = min(s["ece"]   for s in summaries)
    best_auroc = max(s["auroc"] for s in summaries if s["auroc"])

    for s in summaries:
        ece_mark   = "★" if s["ece"]   == best_ece   else " "
        auroc_mark = "★" if s["auroc"] == best_auroc else " "
        gap_str    = f"{s['cal_gap']:+.4f}"
        auroc_str  = f"{s['auroc']:.4f}" if s["auroc"] else "  N/A "

        print(
            f"  {s['method']:<20} "
            f"{s['accuracy']:>9.4f} "
            f"{s['mean_conf']:>9.4f} "
            f"{gap_str:>8} "
            f"{ece_mark}{s['ece']:>7.4f} "
            f"{auroc_mark}{auroc_str:>6} "
            f"{s['mean_sigma']:>10.4f} "
            f"{s['mean_time']:>8.2f}"
        )

    print(f"{'─'*90}")
    print(f"  ★ = best in column")
    print(f"{'═'*90}\n")


# ── Step 4: Plots ─────────────────────────────

PLOT_STYLE = {
    "figure.facecolor":  "#0D1B3E",
    "axes.facecolor":    "#162447",
    "axes.edgecolor":    "#333355",
    "axes.labelcolor":   "#CCD6F6",
    "xtick.color":       "#8892B0",
    "ytick.color":       "#8892B0",
    "text.color":        "#CCD6F6",
    "grid.color":        "#222244",
    "grid.alpha":        0.4,
    "legend.facecolor":  "#0D1B3E",
    "legend.edgecolor":  "#333355",
}

def plot_reliability_diagram(summaries: List[Dict], save_path: str):
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(8, 7))

        # Perfect diagonal
        ax.plot([0,1],[0,1], "--", color="#555577", lw=1.5,
                label="Perfect calibration", zorder=1)
        ax.fill_between([0,1],[0,1],[1,1], alpha=0.04,
                         color="#E05C5C", label="Overconfidence region")

        for s in summaries:
            color    = COLORS[s["method"]]
            bd       = [b for b in s["bin_data"] if b["n"] > 0]
            if not bd: continue
            confs_b  = [b["conf"] for b in bd]
            accs_b   = [b["acc"]  for b in bd]

            ax.plot(confs_b, accs_b, "o-",
                    color=color, lw=2.2, ms=6,
                    label=f"{s['method']}  ECE={s['ece']:.4f}",
                    zorder=3)
            ax.fill_between(confs_b, accs_b, confs_b,
                             alpha=0.06, color=color)

        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Mean Confidence", fontsize=12)
        ax.set_ylabel("Actual Accuracy", fontsize=12)
        ax.set_title("Phase 3 — Reliability Diagram\nAll 4 UQ Methods",
                     fontsize=14, fontweight="bold", color="white")
        ax.grid(True, alpha=0.2)
        ax.legend(loc="upper left", fontsize=9)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor="#0D1B3E")
        plt.close()
    print(f"  Saved → {save_path}")


def plot_ece_bars(summaries: List[Dict], save_path: str):
    with plt.rc_context(PLOT_STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        methods = [s["method"] for s in summaries]
        eces    = [s["ece"]    for s in summaries]
        aurocs  = [s["auroc"] if s["auroc"] else 0 for s in summaries]
        colors  = [COLORS[m] for m in methods]

        # ECE bars
        bars = ax1.bar(methods, eces, color=colors,
                        edgecolor="#0D1B3E", width=0.55)
        ax1.set_title("Expected Calibration Error (↓ better)",
                       fontsize=12, fontweight="bold", color="white")
        ax1.set_ylabel("ECE", color="#CCD6F6")
        ax1.set_xticks(range(len(methods)))
        ax1.set_xticklabels(methods, rotation=15, ha="right", color="#CCD6F6")
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)

        best_ece = min(eces)
        for bar, val in zip(bars, eces):
            marker = " ★" if val == best_ece else ""
            ax1.text(bar.get_x() + bar.get_width()/2,
                      bar.get_height() + 0.002,
                      f"{val:.4f}{marker}",
                      ha="center", va="bottom",
                      color="white", fontsize=9, fontweight="bold")

        # AUROC bars
        bars2 = ax2.bar(methods, aurocs, color=colors,
                         edgecolor="#0D1B3E", width=0.55)
        ax2.set_title("AUROC — confidence predicts correctness (↑ better)",
                       fontsize=12, fontweight="bold", color="white")
        ax2.set_ylabel("AUROC", color="#CCD6F6")
        ax2.set_xticks(range(len(methods)))
        ax2.set_xticklabels(methods, rotation=15, ha="right", color="#CCD6F6")
        ax2.axhline(0.5, color="#888888", linestyle="--", lw=1, label="Random")
        ax2.set_ylim(0, 1.05)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        best_auroc = max(aurocs)
        for bar, val in zip(bars2, aurocs):
            marker = " ★" if val == best_auroc else ""
            ax2.text(bar.get_x() + bar.get_width()/2,
                      bar.get_height() + 0.008,
                      f"{val:.4f}{marker}",
                      ha="center", va="bottom",
                      color="white", fontsize=9, fontweight="bold")

        plt.suptitle("Phase 3 — UQ Method Comparison",
                      fontsize=14, fontweight="bold",
                      color="white", y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor="#0D1B3E")
        plt.close()
    print(f"  Saved → {save_path}")


def plot_per_task_ece(all_results: Dict, save_path: str):
    """Heatmap: rows=methods, cols=tasks, values=ECE."""
    tasks   = ["risk_classification","fraud_detection",
                "premium_estimation","policy_compliance"]
    methods = UQ_METHODS

    matrix = np.zeros((len(methods), len(tasks)))
    for i, method in enumerate(methods):
        breakdown = per_task_metrics(all_results[method])
        for j, task in enumerate(tasks):
            matrix[i, j] = breakdown.get(task, {}).get("ece", 0)

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto",
                        vmin=0, vmax=0.5)

        ax.set_xticks(range(len(tasks)))
        ax.set_xticklabels([t.replace("_", "\n") for t in tasks],
                            color="#CCD6F6", fontsize=10)
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels(methods, color="#CCD6F6", fontsize=10)

        for i in range(len(methods)):
            for j in range(len(tasks)):
                val = matrix[i, j]
                color = "white" if val > 0.25 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                         color=color, fontsize=11, fontweight="bold")

        plt.colorbar(im, ax=ax, label="ECE")
        ax.set_title("Per-Task ECE Heatmap — Phase 3",
                      fontsize=13, fontweight="bold",
                      color="white", pad=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor="#0D1B3E")
        plt.close()
    print(f"  Saved → {save_path}")


def plot_sigma_distributions(all_results: Dict, save_path: str):
    """Shows how σ distributes for correct vs wrong predictions per method."""
    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        for ax, method in zip(axes, UQ_METHODS):
            results = all_results[method]
            color   = COLORS[method]

            # Get σ values from all_uq field (all methods stored per query)
            correct_sigmas = []
            wrong_sigmas   = []
            for r in results:
                uq = r.get("all_uq", {}).get(method, {})
                s  = uq.get("sigma", r.get("sigma", 0))
                if r["is_correct"] is True:
                    correct_sigmas.append(s)
                elif r["is_correct"] is False:
                    wrong_sigmas.append(s)

            bins = np.linspace(0, 1, 20)
            if correct_sigmas:
                ax.hist(correct_sigmas, bins=bins, alpha=0.6,
                         color="#3EC98E", label="Correct", density=True)
            if wrong_sigmas:
                ax.hist(wrong_sigmas, bins=bins, alpha=0.6,
                         color="#E05C5C", label="Wrong", density=True)

            ax.set_title(method, fontsize=10, color=color, fontweight="bold")
            ax.set_xlabel("σ (uncertainty)", color="#8892B0", fontsize=9)
            ax.set_ylabel("Density", color="#8892B0", fontsize=9)
            ax.legend(fontsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # Ideal: correct → low σ, wrong → high σ
            if correct_sigmas and wrong_sigmas:
                c_mean = np.mean(correct_sigmas)
                w_mean = np.mean(wrong_sigmas)
                sep    = w_mean - c_mean
                ax.set_title(f"{method}\n(sep={sep:+.3f})",
                              fontsize=9, color=color)

        plt.suptitle("σ Distribution: Correct vs Wrong Predictions",
                      fontsize=12, fontweight="bold",
                      color="white", y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor="#0D1B3E")
        plt.close()
    print(f"  Saved → {save_path}")


# ── Step 5: Save results ──────────────────────

def save_all_results(all_results: Dict, summaries: List[Dict]):
    # Full results (no hidden states)
    for method, results in all_results.items():
        fname = method.lower().replace(" ", "_")
        path  = f"{RESULTS_DIR}/{fname}_results.json"
        with open(path, "w") as f:
            json.dump(results, f, indent=2, default=str)

    # Summary table
    with open(f"{RESULTS_DIR}/summary.json", "w") as f:
        clean = [{k: v for k, v in s.items() if k != "bin_data"}
                 for s in summaries]
        json.dump(clean, f, indent=2)

    print(f"\n  Results saved to {RESULTS_DIR}/")


# ── Main ──────────────────────────────────────

def main():
    print("\n" + "═"*60)
    print("  PHASE 3 — UQ METHOD COMPARISON")
    print("═"*60)

    api_key  = os.getenv("HF_API_KEY", "")
    backend  = None if _ARGS.backend == "auto" else _ARGS.backend
    use_mock = _ARGS.mock or _ARGS.backend == "mock"

    # If no backend specified and no HF key, auto-detect Ollama/Groq
    if not use_mock and not backend and not api_key:
        from core.hf_api import detect_backend
        auto = detect_backend()
        if auto == "mock":
            print("  ⚠ No real backend detected — running with mock client")
            print("  Use --backend ollama or set GROQ_API_KEY / HF_API_KEY")
            use_mock = True
        else:
            print(f"  Auto-detected backend: {auto}")
            backend = auto
    elif use_mock:
        print("  ⚠ Running with mock client (--mock flag set)")
    elif backend:
        print(f"  Backend: {backend}")
    else:
        print(f"  API key: {api_key[:8]}...")

    # Load data
    test_set = load_test_set()
    print(f"  Test set: {len(test_set)} questions")

    # Limit test size if requested
    if _ARGS.n:
        test_set = test_set[:_ARGS.n]
        print(f"  ⚠ Limited to {_ARGS.n} questions (--n flag)")

    # Run evaluation
    t0          = time.time()
    all_results = run_evaluation(
        test_set, api_key,
        verbose  = True,
        backend  = backend,
        use_mock = use_mock,
    )
    total_time  = time.time() - t0
    print(f"\n  Total evaluation time: {total_time:.0f}s "
          f"({total_time/len(test_set):.1f}s/question)")

    # Compute metrics
    summaries = []
    for method in UQ_METHODS:
        s = compute_metrics(all_results[method], method)
        summaries.append(s)

    # Print table
    print_comparison_table(summaries)

    # Per-task breakdown
    print("\n  Per-task ECE breakdown (VIB Layer):")
    breakdown = per_task_metrics(all_results["VIB Layer"])
    for task, stats in sorted(breakdown.items()):
        print(f"    {task:<28} ECE={stats['ece']:.4f}  "
              f"Acc={stats['accuracy']:.4f}  n={stats['n']}")

    # Generate plots
    print("\n  Generating plots...")
    plot_reliability_diagram(
        summaries,
        f"{RESULTS_DIR}/reliability_diagram.png"
    )
    plot_ece_bars(
        summaries,
        f"{RESULTS_DIR}/ece_comparison.png"
    )
    plot_per_task_ece(
        all_results,
        f"{RESULTS_DIR}/per_task_ece_heatmap.png"
    )
    plot_sigma_distributions(
        all_results,
        f"{RESULTS_DIR}/sigma_distributions.png"
    )

    # Save results
    save_all_results(all_results, summaries)

    print("\n" + "═"*60)
    print("  PHASE 3 COMPLETE")
    print("═"*60)
    print(f"\n  Outputs:")
    for fname in os.listdir(RESULTS_DIR):
        size = os.path.getsize(f"{RESULTS_DIR}/{fname}")
        print(f"    {fname:<45} {size:>8,} bytes")

    # Key findings
    best = min(summaries, key=lambda s: s["ece"])
    print(f"\n  Best calibration: {best['method']} (ECE={best['ece']:.4f})")
    print(f"  Baseline ECE (Softmax): "
          f"{next(s['ece'] for s in summaries if s['method']=='Softmax'):.4f}")
    print(f"\n  These numbers go into your PPT slide 12.")


if __name__ == "__main__":
    main()
