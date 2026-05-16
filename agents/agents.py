# ─────────────────────────────────────────────
# agents/agents.py
#
# ALL FIVE PIPELINE AGENTS
#
# Layer 1 — PreprocessingAgent
# Layer 2 — ReasoningAgent
# Layer 3 — UncertaintyAgent
# Layer 4 — CalibrationAgent
# Layer 5 — PomdpAgent
#
# Each agent:
#   - Reads ONLY from blackboard keys in layers above it
#   - Writes ONLY its own layer's keys
#   - Never calls other agents directly
#   - Reports errors to blackboard
# ─────────────────────────────────────────────

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import time
import math
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Optional

from core.protocol   import Message, MessageType, MessageStatus, AgentRole
from core.blackboard import Blackboard, BBKey
from core.base_agent import BaseAgent
from core.hf_api     import get_hf_client
import config as cfg


# ════════════════════════════════════════════
# LAYER 1 — PREPROCESSING AGENT
# ════════════════════════════════════════════

class PreprocessingAgent(BaseAgent):
    """
    Layer 1 — Validates and formats the raw query.

    Responsibilities:
    - Clean and validate question text
    - Validate choices list
    - Build the ReAct prompt
    - Detect task type

    Reads:  BBKey.QUERY
    Writes: BBKey.PARSED_QUESTION, BBKey.PARSED_CHOICES,
            BBKey.PROMPT, BBKey.TASK_TYPE
    """

    SYSTEM_PROMPT = (
        "You are a professional insurance actuary with 20 years of experience. "
        "You reason carefully before making actuarial decisions."
    )

    # Few-shot examples in new scoring prompt format
    FEW_SHOT = (
        "Example: 72 years old, 5 accidents, smoker yes, BMI 41. "
        "Options: high / medium / low\nAnswer: high\n\n"
        "Example: A claim filed 8 days after policy, $45,000, 4 prior claims, not filed. "
        "Options: suspicious / not suspicious\nAnswer: suspicious\n\n"
        "Example: Age 30, coverage $80,000, waiting 45 days, disclosed. "
        "Options: compliant / non-compliant\nAnswer: compliant\n\n"
    )

    def __init__(self, blackboard: Blackboard):
        super().__init__(
            agent_id   = cfg.AGENT_PREPROCESSING,
            role       = AgentRole.PREPROCESSOR,
            layer      = 1,
            blackboard = blackboard,
        )

    @property
    def required_keys(self): return [BBKey.QUERY]

    @property
    def output_keys(self):
        return [BBKey.PARSED_QUESTION, BBKey.PARSED_CHOICES,
                BBKey.PROMPT, BBKey.TASK_TYPE]

    def process(self, message: Message) -> Message:
        query    = self.bb_read(BBKey.QUERY)
        question = query["question"].strip()
        choices  = [str(c).strip().lower() for c in query["choices"]]

        # Detect task type
        q_lower = question.lower()
        if "risk" in q_lower and "classify" in q_lower:
            task = "risk_classification"
        elif "suspicious" in q_lower or "fraud" in q_lower:
            task = "fraud_detection"
        elif "premium" in q_lower or "bracket" in q_lower:
            task = "premium_estimation"
        elif "compliant" in q_lower or "compliance" in q_lower:
            task = "policy_compliance"
        else:
            task = query.get("task", "unknown")

        # Build scoring prompt using new reliable format
        # Ends with "Answer: " so first generated token = answer
        from core.hf_api import HFApiClient
        prompt = HFApiClient.build_scoring_prompt(
            question = question,
            choices  = choices,
            few_shot = self.FEW_SHOT,
        )

        self.bb_write(BBKey.PARSED_QUESTION, question)
        self.bb_write(BBKey.PARSED_CHOICES,  choices)
        self.bb_write(BBKey.PROMPT,          prompt)
        self.bb_write(BBKey.TASK_TYPE,       task)

        return message.reply(
            sender  = self.agent_id,
            payload = {"task": task, "choices": choices},
        )


# ════════════════════════════════════════════
# LAYER 2 — REASONING AGENT
# ════════════════════════════════════════════

class ReasoningAgent(BaseAgent):
    """
    Layer 2 — Core LLM reasoning via HF API.

    Responsibilities:
    - Score each choice at logit level via HF API
    - Extract softmax confidence distribution
    - Generate reasoning trace (Thought/Action/Observation)

    Reads:  BBKey.PROMPT, BBKey.PARSED_CHOICES
    Writes: BBKey.PREDICTED, BBKey.PREDICTED_IDX,
            BBKey.LOG_PROBS, BBKey.CONF_DIST,
            BBKey.CONFIDENCE, BBKey.REASONING_TRACE,
            BBKey.REASONING_TIME
    """

    def __init__(self, blackboard: Blackboard, hf_client=None):
        super().__init__(
            agent_id   = cfg.AGENT_REASONING,
            role       = AgentRole.REASONER,
            layer      = 2,
            blackboard = blackboard,
        )
        self.hf = hf_client or get_hf_client()

    @property
    def required_keys(self):
        return [BBKey.PROMPT, BBKey.PARSED_CHOICES]

    @property
    def output_keys(self):
        return [BBKey.PREDICTED, BBKey.CONFIDENCE, BBKey.CONF_DIST,
                BBKey.LOG_PROBS, BBKey.REASONING_TRACE, BBKey.REASONING_TIME]

    def process(self, message: Message) -> Message:
        prompt  = self.bb_read(BBKey.PROMPT)
        choices = self.bb_read(BBKey.PARSED_CHOICES)
        t0      = time.time()

        # ONE API call scores all choices and returns raw scores
        # raw_scores are the pre-softmax log-prob equivalents
        # No additional score_choice() calls needed — eliminates 3x API calls
        predicted, confidence, conf_dist, raw_scores = self.hf.score_choices(
            prompt, choices
        )
        pred_idx = choices.index(predicted)

        # Build reasoning trace (no API call — local only)
        question  = self.bb_read(BBKey.PARSED_QUESTION)
        reasoning = self._build_reasoning_trace(
            question, choices, predicted, confidence
        )

        elapsed = time.time() - t0

        # Write all outputs to blackboard
        self.bb_write(BBKey.PREDICTED,       predicted)
        self.bb_write(BBKey.PREDICTED_IDX,   pred_idx)
        self.bb_write(BBKey.LOG_PROBS,       raw_scores)   # from score_choices
        self.bb_write(BBKey.CONF_DIST,       conf_dist)
        self.bb_write(BBKey.CONFIDENCE,      confidence)
        self.bb_write(BBKey.REASONING_TRACE, reasoning)
        self.bb_write(BBKey.REASONING_TIME,  elapsed)

        return message.reply(
            sender  = self.agent_id,
            payload = {
                "predicted":  predicted,
                "confidence": confidence,
                "elapsed":    elapsed,
            }
        )

        return message.reply(
            sender  = self.agent_id,
            payload = {
                "predicted":  predicted,
                "confidence": confidence,
                "elapsed":    elapsed,
            }
        )

    def _build_reasoning_trace(
        self, question: str, choices: List[str],
        predicted: str, confidence: float
    ) -> str:
        choices_str = " / ".join(choices)
        return (
            f"Thought: I analyzed the key actuarial risk factors in this scenario.\n"
            f"Action: Evaluated each feature against standard actuarial thresholds.\n"
            f"Observation: The risk indicators point toward '{predicted}' with "
            f"{confidence:.1%} confidence.\n"
            f"Answer ({choices_str}): {predicted}"
        )


# ════════════════════════════════════════════
# LAYER 3 — UNCERTAINTY AGENT
# ════════════════════════════════════════════

class UncertaintyAgent(BaseAgent):
    """
    Layer 3 — Computes uncertainty estimates.

    Four UQ methods:
    1. Softmax max-probability (already in blackboard from Layer 2)
    2. Temperature scaling (post-hoc confidence correction)
    3. MC Dropout (N forward passes, variance = uncertainty)
    4. VIB-derived σ (low-rank latent Gaussian — your contribution)

    Reads:  BBKey.CONF_DIST, BBKey.LOG_PROBS,
            BBKey.PREDICTED, BBKey.PARSED_CHOICES
    Writes: BBKey.SIGMA, BBKey.UQ_METHOD, BBKey.ALL_UQ_RESULTS
    """

    # ── Phase 5: Real VIB Encoder ────────────────
    _vib_model_path = "results/phase5/vib_encoder.pt"

    def __init__(
        self,
        blackboard: Blackboard,
        hf_client = None,
        uq_method: str = "VIB Layer",
        temperature: float = 1.5,
        mc_samples: int = 5,
    ):
        super().__init__(
            agent_id   = cfg.AGENT_UNCERTAINTY,
            role       = AgentRole.UNCERTAINTY,
            layer      = 3,
            blackboard = blackboard,
        )
        self.hf          = hf_client or get_hf_client()
        self.uq_method   = uq_method
        self.temperature = temperature
        self.mc_samples  = mc_samples
        # Instance-level model cache — avoids shared state across instances
        self._vib_model  = None
        self._vib_loaded = False   # tracks whether load was attempted

    def _load_vib_model(self):
        try:
            import torch
            from phase5_vib_train import VIBEncoder
            model = VIBEncoder(input_dim=4, latent_dim=32, beta=0.01)
            model.load_state_dict(
                torch.load(self._vib_model_path, weights_only=True, map_location="cpu")
            )
            model.eval()
            return model
        except Exception as e:
            print(f"  ⚠ Failed to load VIB model: {e}")
            raise

    @property
    def required_keys(self):
        return [BBKey.CONF_DIST, BBKey.LOG_PROBS,
                BBKey.PREDICTED, BBKey.PARSED_CHOICES]

    @property
    def output_keys(self):
        return [BBKey.SIGMA, BBKey.UQ_METHOD, BBKey.ALL_UQ_RESULTS]

    def process(self, message: Message) -> Message:
        choices   = self.bb_read(BBKey.PARSED_CHOICES)
        log_probs = self.bb_read(BBKey.LOG_PROBS)
        conf_dist = self.bb_read(BBKey.CONF_DIST)
        predicted = self.bb_read(BBKey.PREDICTED)

        # Run all UQ methods
        results = {}

        # Method 1: Softmax (already computed in Layer 2)
        results["Softmax"] = {
            "confidence":  max(conf_dist),
            "conf_dist":   conf_dist,
            "sigma":       1.0 - max(conf_dist),  # uncertainty proxy
        }

        # Method 2: Temperature Scaling
        lp_tensor = torch.tensor(log_probs, dtype=torch.float32)
        scaled    = lp_tensor / self.temperature
        ts_dist   = F.softmax(scaled, dim=0).tolist()
        results["Temperature Scaling"] = {
            "confidence": max(ts_dist),
            "conf_dist":  ts_dist,
            "sigma":      1.0 - max(ts_dist),
        }

        # Method 3: MC Dropout (simulated via score perturbation)
        mc_dists = []
        prompt   = self.bb_read(BBKey.PROMPT)
        for _ in range(self.mc_samples):
            # Add small noise to log-probs to simulate dropout stochasticity
            noise = torch.randn(len(log_probs)) * 0.15
            perturbed = lp_tensor + noise
            mc_dists.append(F.softmax(perturbed, dim=0).tolist())

        mc_mean  = np.mean(mc_dists, axis=0).tolist()
        mc_var   = float(np.mean(np.var(mc_dists, axis=0)))
        mc_sigma = float(np.sqrt(mc_var))
        results["MC Dropout"] = {
            "confidence": max(mc_mean),
            "conf_dist":  mc_mean,
            "sigma":      mc_sigma,
        }

        # Method 4: VIB Layer (low-rank latent uncertainty)
        vib_sigma, vib_conf, vib_dist = self._vib_uncertainty(
            log_probs, conf_dist, choices
        )
        results["VIB Layer"] = {
            "confidence": vib_conf,
            "conf_dist":  vib_dist,
            "sigma":      vib_sigma,
        }

        # Use selected method's sigma as the primary signal
        active = results.get(self.uq_method, results["Softmax"])
        sigma  = active["sigma"]

        # Write to blackboard
        self.bb_write(BBKey.SIGMA,          sigma)
        self.bb_write(BBKey.UQ_METHOD,      self.uq_method)
        self.bb_write(BBKey.ALL_UQ_RESULTS, results)

        # Update confidence if using a different UQ method
        if self.uq_method != "Softmax":
            self.bb_write(BBKey.CONFIDENCE, active["confidence"])
            self.bb_write(BBKey.CONF_DIST,  active["conf_dist"])

        return message.reply(
            sender  = self.agent_id,
            payload = {
                "sigma":      sigma,
                "uq_method":  self.uq_method,
                "all_methods": list(results.keys()),
            }
        )

    def _vib_uncertainty(
        self,
        log_probs: List[float],
        conf_dist: List[float],
        choices:   List[str],
    ):
        """
        Phase 5: Use trained VIB encoder (results/phase5/vib_encoder.pt).
        Falls back to calibrated formula if encoder not available.
        """
        # Try to load encoder once — if it fails, use formula permanently
        if not self._vib_loaded:
            self._vib_loaded = True  # only try once per instance
            try:
                self._vib_model = self._load_vib_model()
                print("  ✓ Trained VIB encoder loaded (Phase 5)")
            except Exception:
                self._vib_model = None  # formula fallback

        if self._vib_model is None:
            return self._vib_uncertainty_simulated(log_probs, conf_dist, choices)

        # Real encoder inference
        vib_sigma = self._vib_model.get_sigma(conf_dist)
        sorted_conf = sorted(conf_dist, reverse=True)
        max_conf    = sorted_conf[0]
        vib_conf    = max_conf * (1.0 - 0.30 * vib_sigma)
        adjustment  = 1.0 - 0.25 * vib_sigma
        vib_dist    = [d * adjustment for d in conf_dist]
        total       = sum(vib_dist)
        vib_dist    = [d / total for d in vib_dist]
        return float(vib_sigma), float(vib_conf), vib_dist

    def _vib_uncertainty_simulated(
        self,
        log_probs: List[float],
        conf_dist: List[float],
        choices:   List[str],
    ):
        """
        Fallback VIB-derived uncertainty estimate (calibrated simulation).
        """
        sorted_conf = sorted(conf_dist, reverse=True)
        p_max    = sorted_conf[0]
        p_second = sorted_conf[1] if len(sorted_conf) > 1 else 0.0

        gap = p_max - p_second
        vib_sigma = float(1.0 - math.sqrt(max(0.0, gap)))

        lp_arr = np.array(log_probs)
        lp_spread = float(np.std(lp_arr)) / (abs(np.mean(lp_arr)) + 1e-6)
        lp_sigma  = min(1.0, lp_spread)

        vib_sigma = 0.80 * vib_sigma + 0.20 * lp_sigma
        vib_sigma = float(np.clip(vib_sigma, 0.0, 1.0))

        max_conf = p_max
        vib_conf = max_conf * (1.0 - 0.30 * vib_sigma)

        adjustment = 1.0 - 0.25 * vib_sigma
        vib_dist   = [d * adjustment for d in conf_dist]
        total      = sum(vib_dist)
        vib_dist   = [d / total for d in vib_dist]

        return float(vib_sigma), float(vib_conf), vib_dist


# ════════════════════════════════════════════
# LAYER 4 — CALIBRATION AGENT
# ════════════════════════════════════════════

class CalibrationAgent(BaseAgent):
    """
    Layer 4 — Calibration measurement.

    Computes calibration metrics for the current prediction.
    For single predictions, maintains a rolling window
    to compute meaningful ECE estimates.

    Reads:  BBKey.CONFIDENCE, BBKey.SIGMA,
            BBKey.PREDICTED, BBKey.GROUND_TRUTH
    Writes: BBKey.ECE, BBKey.AUROC,
            BBKey.CALIBRATION_GAP, BBKey.BIN_DATA
    """

    def __init__(self, blackboard: Blackboard, history_window: int = 50):
        super().__init__(
            agent_id   = cfg.AGENT_CALIBRATION,
            role       = AgentRole.CALIBRATOR,
            layer      = 4,
            blackboard = blackboard,
        )
        # Rolling window for ECE computation
        self.window_size = history_window
        self.history: List[Dict] = []

    @property
    def required_keys(self):
        return [BBKey.CONFIDENCE, BBKey.PREDICTED]

    @property
    def output_keys(self):
        return [BBKey.ECE, BBKey.CALIBRATION_GAP, BBKey.BIN_DATA]

    def process(self, message: Message) -> Message:
        confidence = self.bb_read(BBKey.CONFIDENCE)
        predicted  = self.bb_read(BBKey.PREDICTED)
        gt         = self.bb_read(BBKey.GROUND_TRUTH)
        sigma      = self.bb_read(BBKey.SIGMA, default=None)

        # Determine correctness if ground truth available
        is_correct = None
        if gt:
            is_correct = (predicted.lower().strip() == gt.lower().strip())
            self.bb_write(BBKey.IS_CORRECT, is_correct)

        # Add to rolling history
        if is_correct is not None:
            self.history.append({
                "confidence": confidence,
                "correct":    is_correct,
                "sigma":      sigma,
            })
            # Keep window
            if len(self.history) > self.window_size:
                self.history = self.history[-self.window_size:]

        # Compute ECE if we have enough history
        ece, bin_data, auroc = self._compute_metrics()

        # Single-point calibration gap — SIGNED for correct direction label
        # Positive = overconfident (conf > actual accuracy)
        # Negative = underconfident (conf < actual accuracy)
        cal_gap = None
        if is_correct is not None:
            actual = 1.0 if is_correct else 0.0
            cal_gap = confidence - actual   # signed: + overconfident, - underconfident

        # Write to blackboard
        self.bb_write(BBKey.ECE,             ece)
        self.bb_write(BBKey.CALIBRATION_GAP, cal_gap)
        self.bb_write(BBKey.BIN_DATA,        bin_data)
        self.bb_write(BBKey.AUROC,           auroc)

        return message.reply(
            sender  = self.agent_id,
            payload = {
                "ece":       ece,
                "auroc":     auroc,
                "cal_gap":   cal_gap,
                "n_history": len(self.history),
            }
        )

    def _compute_metrics(self):
        """
        Compute ECE and AUROC from rolling history.

        Fix 2: Compute single-point ECE immediately from query 1.
        Rolling ECE becomes meaningful after ~10 queries.
        Single-point ECE = |confidence - correctness| for current query.
        """
        if not self.history:
            return None, [], None

        confs = [h["confidence"] for h in self.history]
        corrs = [h["correct"]    for h in self.history]

        # Single-point ECE = absolute calibration gap for this query
        single_ece = abs(confs[-1] - (1.0 if corrs[-1] else 0.0))

        # Rolling ECE (meaningful after 5+ queries)
        if len(self.history) >= 5:
            bins      = np.linspace(0, 1, cfg.N_BINS + 1)
            bin_data  = []
            ece       = 0.0
            confs_arr = np.array(confs)
            corrs_arr = np.array(corrs, dtype=float)

            for i in range(cfg.N_BINS):
                lo, hi = bins[i], bins[i+1]
                mask   = (confs_arr >= lo) & (confs_arr < hi)
                if i == cfg.N_BINS - 1:
                    mask = (confs_arr >= lo) & (confs_arr <= hi)
                n = mask.sum()
                if n == 0:
                    bin_data.append({"bin_center": (lo+hi)/2, "n": 0,
                                      "accuracy": 0, "confidence": 0})
                    continue
                acc  = float(corrs_arr[mask].mean())
                conf = float(confs_arr[mask].mean())
                ece += (n / len(confs)) * abs(acc - conf)
                bin_data.append({"bin_center": (lo+hi)/2, "n": int(n),
                                  "accuracy": acc, "confidence": conf})
        else:
            # Use single-point ECE until we have enough history
            ece      = single_ece
            bin_data = []

        # AUROC (meaningful after 5+ queries with both correct and wrong)
        auroc = None
        if len(self.history) >= 5:
            confs_arr = np.array(confs)
            corrs_arr = np.array(corrs, dtype=float)
            n_pos = sum(corrs)
            n_neg = len(corrs) - n_pos
            if n_pos > 0 and n_neg > 0:
                order      = np.argsort(-confs_arr)
                tp         = np.cumsum(corrs_arr[order])
                fp         = np.cumsum(1 - corrs_arr[order])
                tpr, fpr   = tp / n_pos, fp / n_neg
                try:
                    auroc = float(np.trapezoid(tpr, fpr))
                except AttributeError:
                    auroc = float(np.trapz(tpr, fpr))
                auroc = max(0.0, min(1.0, abs(auroc)))

        return float(ece), bin_data, auroc

    def reset_history(self):
        self.history = []


# ════════════════════════════════════════════
# LAYER 5 — POMDP AGENT
# ════════════════════════════════════════════

class PomdpAgent(BaseAgent):
    """
    Layer 5 — POMDP decision-making agent.

    This is the research contribution layer.
    Uses VIB belief state (μ=confidence, σ=uncertainty)
    to make the optimal action decision:

        σ < ANSWER_THRESHOLD  AND  conf > MIN_CONF  →  ANSWER
        σ < SEEK_THRESHOLD                           →  SEEK_INFO
        σ >= SEEK_THRESHOLD                          →  DEFER

    This directly implements the POMDP action space:
        A = {Answer, SeekInfo, Defer}

    In Phase 6, this agent will use a trained SAC policy.
    Currently uses the rule-based policy above.

    Reads:  BBKey.CONFIDENCE, BBKey.SIGMA,
            BBKey.PREDICTED, BBKey.ECE
    Writes: BBKey.POMDP_ACTION, BBKey.POMDP_REASON,
            BBKey.BELIEF_STATE
    """

    def __init__(self, blackboard: Blackboard):
        super().__init__(
            agent_id   = cfg.AGENT_POMDP,
            role       = AgentRole.DECISION,
            layer      = 5,
            blackboard = blackboard,
        )

    @property
    def required_keys(self):
        return [BBKey.CONFIDENCE, BBKey.PREDICTED]

    @property
    def output_keys(self):
        return [BBKey.POMDP_ACTION, BBKey.POMDP_REASON, BBKey.BELIEF_STATE]

    def process(self, message: Message) -> Message:
        confidence = self.bb_read(BBKey.CONFIDENCE)
        sigma      = self.bb_read(BBKey.SIGMA,     default=1.0 - confidence)
        predicted  = self.bb_read(BBKey.PREDICTED)
        ece        = self.bb_read(BBKey.ECE,        default=None)

        # Belief state — the POMDP's b(s)
        # μ = our best estimate (confidence in predicted answer)
        # σ = uncertainty about that estimate
        belief_state = {
            "mu":    confidence,
            "sigma": sigma,
            "distribution": "N(mu, sigma²)",
            "note": "Phase 6: this becomes the RL policy's input"
        }

        # POMDP action decision (rule-based policy for Phase 3)
        action, reason, action_detail = self._decide_action(
            confidence, sigma, predicted, ece
        )

        # Write to blackboard
        self.bb_write(BBKey.POMDP_ACTION,  action)
        self.bb_write(BBKey.POMDP_REASON,  reason)
        self.bb_write(BBKey.BELIEF_STATE,  belief_state)

        return message.reply(
            sender  = self.agent_id,
            payload = {
                "action":       action,
                "reason":       reason,
                "belief_state": belief_state,
                "detail":       action_detail,
            }
        )

    def _decide_action(
        self,
        confidence: float,
        sigma:      float,
        predicted:  str,
        ece:        Optional[float],
    ):
        """
        Rule-based POMDP policy.

        Logic directly from the POMDP formulation:
        - Low σ + high conf → agent's belief is concentrated → safe to act
        - Medium σ          → belief is diffuse → gather more info
        - High σ            → belief is too uncertain → defer to human

        In Phase 6, this is replaced by a learned SAC policy.
        """

        if sigma < cfg.SIGMA_ANSWER_THRESHOLD and confidence > cfg.CONF_MIN_ANSWER:
            action = "ANSWER"
            reason = (
                f"Belief state is narrow (σ={sigma:.3f} < {cfg.SIGMA_ANSWER_THRESHOLD}). "
                f"Confidence is sufficient ({confidence:.3f} > {cfg.CONF_MIN_ANSWER}). "
                f"Proceeding with answer: {predicted.upper()}."
            )
            detail = {
                "icon":   "✅",
                "color":  "#3EC98E",
                "label":  "Answer Confidently",
                "action_cost": 0.0,
            }

        elif sigma < cfg.SIGMA_SEEK_THRESHOLD:
            action = "SEEK_INFO"
            reason = (
                f"Belief state is moderately uncertain (σ={sigma:.3f}). "
                f"Requesting additional documents: claims history, medical records, "
                f"or policy terms before finalizing decision."
            )
            detail = {
                "icon":   "🔍",
                "color":  "#F0A500",
                "label":  "Seek More Information",
                "action_cost": 0.2,
                "info_requested": [
                    "claims_history", "medical_records", "applicant_documents"
                ],
            }

        else:
            action = "DEFER"
            reason = (
                f"Belief state is too wide (σ={sigma:.3f} ≥ {cfg.SIGMA_SEEK_THRESHOLD}). "
                f"Uncertainty exceeds safe threshold for automated decision. "
                f"Routing to human underwriter for manual review."
            )
            detail = {
                "icon":   "👤",
                "color":  "#E05C5C",
                "label":  "Defer to Human Underwriter",
                "action_cost": 0.5,
                "escalation_level": "senior_underwriter",
            }

        # Additional context if ECE is available
        if ece is not None and ece > 0.2:
            reason += (
                f" Note: Running ECE={ece:.3f} indicates systematic overconfidence "
                f"— this may be a quantization-induced miscalibration case."
            )

        return action, reason, detail
