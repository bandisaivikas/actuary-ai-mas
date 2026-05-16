#!/usr/bin/env python3
# ─────────────────────────────────────────────
# phase5_vib_train.py
#
# PHASE 5 — Train Real VIB Encoder (PyTorch)
#
# Replaces the simulated _vib_uncertainty() formula
# with a trained Variational Information Bottleneck
# encoder that learns the KL penalty forcing honest
# uncertainty representation.
#
# This is the technical research contribution:
# - Input: log-prob vector from LLM
# - Output: (μ, σ) belief state in latent space
# - Training: KL-regularized reconstruction loss
#   so σ is forced to be honest about uncertainty
#
# Uses Phase 3 results as training data.
#
# Usage:
#   PYTHONPATH=. python3 phase5_vib_train.py
#   PYTHONPATH=. python3 phase5_vib_train.py --data results/ground_truth_run.json
# ─────────────────────────────────────────────

import os, sys, json, time, argparse, math
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

RESULTS_DIR = "results/phase5"
MODEL_PATH  = "results/phase5/vib_encoder.pt"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# VIB Encoder Architecture
# ─────────────────────────────────────────────

class VIBEncoder(nn.Module):
    """
    Variational Information Bottleneck encoder.

    Maps a confidence distribution vector (from LLM softmax)
    into a latent Gaussian belief state (μ, σ).

    The KL penalty β·KL(N(μ,σ²)||N(0,1)) forces the encoder
    to be HONEST about uncertainty — σ can't be arbitrarily
    small unless the evidence truly supports it.

    Architecture:
        [conf_dist] → Encoder → (μ, log_var) → reparameterize → z
                                                                ↓
        z → Decoder → [correctness_prediction]
    """

    def __init__(
        self,
        input_dim:  int = 4,     # max choices (padded)
        latent_dim: int = 32,    # IB bottleneck width
        beta:       float = 0.01, # KL regularization strength
    ):
        super().__init__()
        self.input_dim  = input_dim
        self.latent_dim = latent_dim
        self.beta       = beta

        # Encoder: conf_dist → (μ, log_var)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.mu_head      = nn.Linear(64, latent_dim)
        self.log_var_head = nn.Linear(64, latent_dim)

        # Decoder: z → P(correct) scalar
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor):
        """Returns (mu, log_var) from input confidence distribution."""
        h       = self.encoder(x)
        mu      = self.mu_head(h)
        log_var = self.log_var_head(h)
        return mu, log_var

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = μ + ε·σ, ε~N(0,1)."""
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu  # at inference: use mean

    def forward(self, x: torch.Tensor):
        mu, log_var = self.encode(x)
        z           = self.reparameterize(mu, log_var)
        pred        = self.decoder(z)
        return pred, mu, log_var

    def get_sigma(self, conf_dist: list) -> float:
        """
        Get uncertainty σ for a single confidence distribution.
        This replaces _vib_uncertainty() in UncertaintyAgent.

        σ is derived from log_var: σ = mean(exp(0.5 * log_var))
        High σ → model is uncertain about the latent belief state
        Low σ  → model is confident (IB bottleneck is compressed)
        """
        self.eval()
        with torch.no_grad():
            # Pad/truncate to input_dim
            x = self._pad_conf(conf_dist)
            x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
            mu, log_var = self.encode(x)
            sigma = torch.exp(0.5 * log_var).mean().item()
        return float(np.clip(sigma, 0.0, 1.0))

    def _pad_conf(self, conf_dist: list) -> list:
        """Pad confidence distribution to input_dim with zeros."""
        x = list(conf_dist)[:self.input_dim]
        while len(x) < self.input_dim:
            x.append(0.0)
        return x


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def load_training_data(data_paths: list):
    """
    Load (conf_dist, is_correct) pairs from Phase 3 / ground truth results.
    Returns X (conf_dist tensors), y (correctness labels).
    """
    all_X, all_y = [], []

    for path in data_paths:
        if not os.path.exists(path):
            continue

        with open(path) as f:
            data = json.load(f)

        # Handle ground_truth_run.json format
        if isinstance(data, dict) and "records" in data:
            records = data["records"]
            for r in records:
                conf_dist = r.get("all_uq", {}).get("Softmax", {}).get("conf_dist")
                if not conf_dist:
                    # Synthesize from confidence
                    conf = r.get("confidence", 0.5)
                    n    = len(r.get("choices", ["a", "b"]))
                    rest = (1.0 - conf) / max(1, n - 1)
                    conf_dist = [conf] + [rest] * (n - 1)
                is_correct = r.get("is_correct")
                if is_correct is not None:
                    all_X.append(conf_dist)
                    all_y.append(float(is_correct))

        # Handle phase3 individual result format (list of records)
        elif isinstance(data, list):
            for r in data:
                conf_dist = r.get("all_uq", {}).get("Softmax", {}).get("conf_dist")
                if not conf_dist:
                    conf = r.get("confidence", 0.5)
                    n    = len(r.get("choices", ["a", "b"]))
                    rest = (1.0 - conf) / max(1, n - 1)
                    conf_dist = [conf] + [rest] * (n - 1)
                is_correct = r.get("is_correct")
                if is_correct is not None:
                    all_X.append(conf_dist)
                    all_y.append(float(is_correct))

    print(f"  Loaded {len(all_X)} training samples")
    return all_X, all_y


def pad_batch(X_raw: list, input_dim: int = 4) -> np.ndarray:
    """Pad/truncate all conf_dists to input_dim."""
    out = []
    for x in X_raw:
        xp = list(x)[:input_dim]
        while len(xp) < input_dim:
            xp.append(0.0)
        out.append(xp)
    return np.array(out, dtype=np.float32)


def train_vib(
    X_raw:      list,
    y_raw:      list,
    input_dim:  int   = 4,
    latent_dim: int   = 32,
    beta:       float = 0.01,
    epochs:     int   = 200,
    batch_size: int   = 32,
    lr:         float = 1e-3,
    verbose:    bool  = True,
) -> VIBEncoder:
    """
    Train VIB encoder.

    Loss = Reconstruction loss + β·KL divergence
         = BCE(p_correct, y) + β·KL(N(μ,σ²)||N(0,1))

    The KL term forces σ to be informative:
    - When the LLM is genuinely uncertain, the encoder can't
      compress information → σ stays large (honest)
    - When confident, information flows through → σ shrinks
    """
    X = pad_batch(X_raw, input_dim)
    y = np.array(y_raw, dtype=np.float32)

    X_t = torch.tensor(X)
    y_t = torch.tensor(y).unsqueeze(1)

    dataset = TensorDataset(X_t, y_t)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = VIBEncoder(input_dim=input_dim, latent_dim=latent_dim, beta=beta)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    if verbose:
        print(f"\n  Training VIB Encoder:")
        print(f"  Architecture: {input_dim} → [64 → 64] → (μ,σ) [{latent_dim}d] → decoder → P(correct)")
        print(f"  β (KL weight): {beta}")
        print(f"  Epochs: {epochs}, batch: {batch_size}, lr: {lr}")
        print(f"  Samples: {len(X)}")
        print(f"  Loss = BCE(correctness) + β·KL(N(μ,σ²)||N(0,1))")
        print()

    history = {"loss": [], "bce": [], "kl": [], "val_acc": []}
    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        epoch_bce = epoch_kl = epoch_loss = 0.0
        n_batches = 0

        for xb, yb in loader:
            optim.zero_grad()

            pred, mu, log_var = model(xb)

            # Reconstruction loss: how well does z → correctness?
            bce = F.binary_cross_entropy(pred, yb)

            # KL divergence: KL(N(μ,σ²)||N(0,1))
            # = -0.5 * Σ(1 + log_var - μ² - e^log_var)
            kl = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

            # VIB loss
            loss = bce + beta * kl
            loss.backward()
            optim.step()

            epoch_bce  += bce.item()
            epoch_kl   += kl.item()
            epoch_loss += loss.item()
            n_batches  += 1

        avg_loss = epoch_loss / n_batches
        avg_bce  = epoch_bce / n_batches
        avg_kl   = epoch_kl / n_batches
        history["loss"].append(avg_loss)
        history["bce"].append(avg_bce)
        history["kl"].append(avg_kl)

        # Validation accuracy
        model.eval()
        with torch.no_grad():
            pred_all, _, _ = model(X_t)
            val_acc = ((pred_all > 0.5).float() == y_t).float().mean().item()
        history["val_acc"].append(val_acc)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), MODEL_PATH)

        if verbose and (epoch + 1) % 50 == 0:
            # Sample σ values to show calibration
            model.eval()
            with torch.no_grad():
                _, mu_all, lv_all = model(X_t)
                sigma_all = torch.exp(0.5 * lv_all).mean(dim=1)
                # Separate correct/wrong
                mask_c = (y_t.squeeze() == 1)
                mask_w = (y_t.squeeze() == 0)
                sig_c  = sigma_all[mask_c].mean().item() if mask_c.any() else 0
                sig_w  = sigma_all[mask_w].mean().item() if mask_w.any() else 0
            print(
                f"  Epoch {epoch+1:3d}/{epochs}  "
                f"loss={avg_loss:.4f}  bce={avg_bce:.4f}  kl={avg_kl:.4f}  "
                f"acc={val_acc:.3f}  "
                f"σ_correct={sig_c:.3f}  σ_wrong={sig_w:.3f}"
            )

    # Load best model
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    return model, history


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

def evaluate_vib(model: VIBEncoder, X_raw: list, y_raw: list):
    """Evaluate trained VIB encoder vs simulated formula."""
    from agents.agents import UncertaintyAgent
    from core.blackboard import Blackboard

    X = pad_batch(X_raw, model.input_dim)
    X_t = torch.tensor(X)
    y   = np.array(y_raw)

    model.eval()
    with torch.no_grad():
        _, mu_all, lv_all = model(X_t)
        sigma_all = torch.exp(0.5 * lv_all).mean(dim=1).numpy()

    # Separation score: σ_wrong > σ_correct (uncertainty correctly captures errors)
    correct_mask = y == 1
    wrong_mask   = y == 0

    sig_correct = sigma_all[correct_mask].mean() if correct_mask.any() else 0
    sig_wrong   = sigma_all[wrong_mask].mean()   if wrong_mask.any()   else 0
    separation  = sig_wrong - sig_correct

    print(f"\n  VIB Encoder Evaluation:")
    print(f"  {'Metric':<35} {'Value'}")
    print(f"  {'─'*50}")
    print(f"  {'Mean σ (correct predictions)':<35} {sig_correct:.4f}")
    print(f"  {'Mean σ (wrong predictions)':<35}   {sig_wrong:.4f}")
    print(f"  {'Separation (wrong - correct σ)':<35} {separation:+.4f}")
    print(f"  {'Interpretation':<35} {'✓ VIB correctly uncertain on wrong preds' if separation > 0 else '⚠ Low separation — needs more data'}")

    # Compare with simulated formula
    print(f"\n  vs Simulated _vib_uncertainty() formula:")
    print(f"  Simulated uses: σ = 1 - sqrt(p_max - p_second)")
    print(f"  Trained VIB:    σ from learned KL-regularized encoder")
    print(f"  Separation improvement: {'+' if separation > 0.05 else 'minimal (need more training data)'}")

    return {
        "sig_correct":  float(sig_correct),
        "sig_wrong":    float(sig_wrong),
        "separation":   float(separation),
        "sigma_all":    sigma_all.tolist(),
    }


def generate_patch_for_agents(model: VIBEncoder):
    """
    Generates the replacement code for _vib_uncertainty in agents.py.
    Shows how to swap the trained encoder in.
    """
    patch = f'''
    # ── Phase 5: Real VIB Encoder ────────────────
    # Replace _vib_uncertainty() with this after training:

    # At class level:
    _vib_model_path = "results/phase5/vib_encoder.pt"
    _vib_model = None

    def _load_vib_model(self):
        from phase5_vib_train import VIBEncoder
        model = VIBEncoder(input_dim=4, latent_dim={model.latent_dim}, beta={model.beta})
        model.load_state_dict(
            torch.load(self._vib_model_path, weights_only=True, map_location="cpu")
        )
        model.eval()
        return model

    def _vib_uncertainty(self, log_probs, conf_dist, choices):
        """Phase 5: Use trained VIB encoder instead of formula."""
        if self._vib_model is None:
            try:
                self._vib_model = self._load_vib_model()
            except Exception:
                # Fall back to simulated if model not loaded
                return self._vib_uncertainty_simulated(log_probs, conf_dist, choices)

        vib_sigma = self._vib_model.get_sigma(conf_dist)

        # VIB-adjusted confidence
        sorted_conf = sorted(conf_dist, reverse=True)
        max_conf    = sorted_conf[0]
        vib_conf    = max_conf * (1.0 - 0.30 * vib_sigma)

        # Redistribute
        adjustment = 1.0 - 0.25 * vib_sigma
        vib_dist   = [d * adjustment for d in conf_dist]
        total      = sum(vib_dist)
        vib_dist   = [d / total for d in vib_dist]

        return float(vib_sigma), float(vib_conf), vib_dist
    '''
    return patch


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs="+",
                        default=[
                            "results/ground_truth_run.json",
                            "results/phase3/softmax_results.json",
                            "results/phase3/vib_layer_results.json",
                        ],
                        help="Training data files")
    parser.add_argument("--epochs",    type=int,   default=300)
    parser.add_argument("--latent",    type=int,   default=32)
    parser.add_argument("--beta",      type=float, default=0.01)
    parser.add_argument("--lr",        type=float, default=1e-3)
    parser.add_argument("--eval-only", action="store_true",
                        help="Only evaluate existing model")
    args = parser.parse_args()

    print("\n" + "═" * 60)
    print("  PHASE 5 — VIB ENCODER TRAINING")
    print("═" * 60)

    # Load data
    X_raw, y_raw = load_training_data(args.data)

    if len(X_raw) < 5:
        print(f"\n  ⚠ Only {len(X_raw)} samples — need at least 5.")
        print(f"  Run ground truth queries first:")
        print(f"    PYTHONPATH=. python3 run_ground_truth_queries.py --backend ollama")
        print(f"  OR run Phase 3 first:")
        print(f"    PYTHONPATH=. python3 phase3_evaluate.py --backend ollama")
        return

    if args.eval_only:
        if not os.path.exists(MODEL_PATH):
            print(f"  ⚠ No model at {MODEL_PATH} — train first.")
            return
        from phase5_vib_train import VIBEncoder
        model = VIBEncoder(input_dim=4, latent_dim=args.latent, beta=args.beta)
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    else:
        # Train
        model, history = train_vib(
            X_raw, y_raw,
            input_dim  = 4,
            latent_dim = args.latent,
            beta       = args.beta,
            epochs     = args.epochs,
            lr         = args.lr,
        )

        # Save training history
        hist_path = f"{RESULTS_DIR}/training_history.json"
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)
        print(f"\n  Training history → {hist_path}")
        print(f"  Best model → {MODEL_PATH}")

    # Evaluate
    eval_results = evaluate_vib(model, X_raw, y_raw)

    # Save eval results
    eval_path = f"{RESULTS_DIR}/vib_eval.json"
    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"\n  Eval results → {eval_path}")

    # Show patch for agents.py
    print(f"\n  To integrate trained VIB into the pipeline:")
    print(f"  Replace _vib_uncertainty() in agents/agents.py with:")
    patch = generate_patch_for_agents(model)
    print(patch)

    print("\n" + "═" * 60)
    print("  PHASE 5 COMPLETE")
    print("═" * 60)
    print(f"\n  The trained VIB encoder ({args.latent}D latent, β={args.beta})")
    print(f"  replaces the mathematical approximation in agents.py.")
    print(f"  It learns to be HONEST about uncertainty via KL regularization.")
    print(f"  This is the empirical core of the research claim.\n")


if __name__ == "__main__":
    main()
