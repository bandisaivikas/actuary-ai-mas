#!/usr/bin/env python3
# ─────────────────────────────────────────────
# prewarm_demo.py
#
# PRIORITY 3 — Pre-run 10 queries before demo day
#
# Runs 10 queries through the pipeline and saves
# calibration history to disk so the demo starts
# with rolling ECE already active (not cold).
#
# CalibrationAgent.history is in-memory only and
# resets on every Streamlit restart. This script
# persists it to results/demo_history.json so
# demo_app.py can reload it on startup.
#
# Usage (night before demo):
#   PYTHONPATH=. python prewarm_demo.py
#
# Then launch demo:
#   streamlit run demo_app.py
# ─────────────────────────────────────────────

import os, sys, json, time, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from core.orchestrator import Orchestrator
import config as cfg

HISTORY_PATH = "results/demo_history.json"

# 10 diverse queries for prewarming (ground truth included)
PREWARM_QUERIES = [
    {"question": "An insurance applicant is 68 years old, has had 4 accidents in the past 5 years, filed 3 claims, is a smoker: yes, and has a BMI of 39.2. Classify the risk level.", "choices": ["high", "medium", "low"], "answer": "high", "task": "risk_classification"},
    {"question": "An insurance applicant is 26 years old, has had 0 accidents in the past 5 years, filed 0 claims, is a smoker: no, and has a BMI of 22.1. Classify the risk level.", "choices": ["high", "medium", "low"], "answer": "low", "task": "risk_classification"},
    {"question": "An insurance applicant is 55 years old, has had 2 accidents in the past 5 years, filed 1 claim, is a smoker: no, and has a BMI of 31.5. Classify the risk level.", "choices": ["high", "medium", "low"], "answer": "medium", "task": "risk_classification"},
    {"question": "A claim was filed 8 days after policy activation. Claimed amount: $48,000. Prior claims: 5. Document inconsistencies found: 3. Police report: not filed. Is this claim suspicious or not suspicious?", "choices": ["suspicious", "not suspicious"], "answer": "suspicious", "task": "fraud_detection"},
    {"question": "A claim was filed 380 days after policy activation. Claimed amount: $2,800. Prior claims: 0. Document inconsistencies found: 0. Police report: filed. Is this claim suspicious or not suspicious?", "choices": ["suspicious", "not suspicious"], "answer": "not suspicious", "task": "fraud_detection"},
    {"question": "Policy requires minimum age 25, maximum coverage $100,000, waiting period of 60 days. Applicant: age 22, requested coverage $120,000, waiting period 15 days. Is this application compliant or non-compliant?", "choices": ["compliant", "non-compliant"], "answer": "non-compliant", "task": "policy_compliance"},
    {"question": "Policy requires minimum age 18, maximum coverage $150,000, waiting period of 30 days. Applicant: age 35, requested coverage $80,000, waiting period 45 days. Is this application compliant or non-compliant?", "choices": ["compliant", "non-compliant"], "answer": "compliant", "task": "policy_compliance"},
    {"question": "An insurance applicant is 75 years old, has had 6 accidents in the past 5 years, filed 5 claims, is a smoker: yes, and has a BMI of 44.0. Classify the risk level.", "choices": ["high", "medium", "low"], "answer": "high", "task": "risk_classification"},
    {"question": "A claim was filed 15 days after policy activation. Claimed amount: $32,000. Prior claims: 3. Document inconsistencies found: 2. Police report: not filed. Is this claim suspicious or not suspicious?", "choices": ["suspicious", "not suspicious"], "answer": "suspicious", "task": "fraud_detection"},
    {"question": "An applicant is 28 years old, smoker: no, 0 accidents, BMI 21.5, coverage $100,000. Into which premium bracket does this applicant fall?", "choices": ["high", "medium", "low"], "answer": "low", "task": "premium_estimation"},
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="auto",
                        choices=["auto", "ollama", "groq", "hf", "mock"])
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--uq", default="VIB Layer",
                        choices=["Softmax", "Temperature Scaling", "MC Dropout", "VIB Layer"])
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    print("\n" + "═" * 55)
    print("  DEMO PRE-WARM — Running 10 seeding queries")
    print("═" * 55)

    use_mock   = args.mock or args.backend == "mock"
    backend_kw = None if args.backend == "auto" else args.backend
    api_key    = os.getenv("HF_API_KEY", "")

    orch = Orchestrator(
        api_key    = api_key,
        model_name = cfg.HF_MODEL_SMALL,
        uq_method  = args.uq,
        use_mock   = use_mock,
        backend    = backend_kw,
    )

    history = []
    for i, q in enumerate(PREWARM_QUERIES):
        print(f"  Query {i+1}/{len(PREWARM_QUERIES)}: {q['task'][:25]}...", end=" ", flush=True)
        t0 = time.time()
        result = orch.run(
            question     = q["question"],
            choices      = q["choices"],
            ground_truth = q["answer"],
            task         = q["task"],
            verbose      = False,
        )
        elapsed = time.time() - t0

        conf      = result.get("confidence") or 0.0
        sigma     = result.get("sigma") or 0.0
        is_correct = result.get("is_correct")

        if is_correct is not None:
            history.append({
                "confidence": conf,
                "correct":    is_correct,
                "sigma":      sigma,
            })

        ok = "✓" if is_correct else ("✗" if is_correct is False else "?")
        print(f"{ok} conf={conf:.3f} σ={sigma:.3f} ({elapsed:.1f}s)")

    # Compute ECE from history
    if len(history) >= 5:
        confs = np.array([h["confidence"] for h in history])
        corrs = np.array([h["correct"] for h in history], dtype=float)
        bins  = np.linspace(0, 1, cfg.N_BINS + 1)
        ece   = 0.0
        for i in range(cfg.N_BINS):
            lo, hi = bins[i], bins[i+1]
            mask   = (confs >= lo) & (confs < hi)
            if i == cfg.N_BINS - 1:
                mask = (confs >= lo) & (confs <= hi)
            n = mask.sum()
            if n == 0:
                continue
            ece += (n / len(confs)) * abs(corrs[mask].mean() - confs[mask].mean())
        print(f"\n  Pre-warm ECE: {ece:.4f} (rolling, n={len(history)})")
    else:
        print(f"\n  {len(history)} ground-truth queries recorded")

    # Save to disk
    payload = {
        "history":    history,
        "n_queries":  len(history),
        "uq_method":  args.uq,
        "backend":    args.backend,
        "timestamp":  time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(HISTORY_PATH, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\n  History saved → {HISTORY_PATH}")
    print(f"  Demo will load this on startup — ECE rolling from query 1!\n")
    print(f"  Now launch: streamlit run demo_app.py\n")


if __name__ == "__main__":
    main()
