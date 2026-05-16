# ─────────────────────────────────────────────
# core/orchestrator.py
#
# ORCHESTRATOR — Central Pipeline Controller
#
# The orchestrator:
#   1. Receives every incoming query
#   2. Resets and initializes the blackboard
#   3. Routes messages through the 5-layer pipeline
#   4. Handles agent failures gracefully
#   5. Collects and returns the final result
#   6. Maintains the message log for demo display
#
# Communication modes:
#   - Centralized sequential (default)
#   - Parallel Layer 3+4 (uncertainty + calibration run together)
#   - Reactive (subscribes to blackboard events)
# ─────────────────────────────────────────────

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import time
import threading
from typing import Dict, List, Optional, Any
from datetime import datetime

from core.protocol   import (
    Message, MessageType, MessageStatus, AgentRole,
    make_query_message
)
from core.blackboard  import Blackboard, BBKey
from core.hf_api      import get_hf_client
from agents.agents    import (
    PreprocessingAgent, ReasoningAgent,
    UncertaintyAgent, CalibrationAgent, PomdpAgent
)
import config as cfg


class Orchestrator:
    """
    Central coordinator for the 5-layer MAS pipeline.

    Execution model:
        Sequential layers 1 → 2 → 3 → 4 → 5
        Each layer reads from blackboard, executes, writes back.
        Orchestrator checks layer completion before triggering next.

    Failure handling:
        If any layer fails, orchestrator:
        1. Logs the error to blackboard
        2. Attempts fallback (where possible)
        3. Continues pipeline with available data
        4. Returns partial result rather than crashing

    Demo features:
        - message_log: full trace of all inter-agent messages
        - pipeline_events: timestamped event stream for live display
        - blackboard_snapshots: BB state after each layer
    """

    def __init__(
        self,
        api_key:    Optional[str] = None,
        model_name: str = cfg.HF_MODEL_SMALL,
        uq_method:  str = "VIB Layer",
        use_mock:   bool = False,
        backend:    Optional[str] = None,
    ):
        # Shared blackboard
        self.bb = Blackboard()

        # Inference client — auto-detects Ollama > Groq > HF > Mock
        self.hf = get_hf_client(
            api_key    = api_key,
            model_name = model_name,
            use_mock   = use_mock,
            backend    = backend,
        )

        # Initialize all agents
        self.agents = {
            "preprocessing": PreprocessingAgent(self.bb),
            "reasoning":     ReasoningAgent(self.bb, self.hf),
            "uncertainty":   UncertaintyAgent(self.bb, self.hf, uq_method),
            "calibration":   CalibrationAgent(self.bb),
            "pomdp":         PomdpAgent(self.bb),
        }

        # Communication log
        self.message_log:          List[Dict] = []
        self.pipeline_events:      List[Dict] = []
        self.blackboard_snapshots: Dict[int, Dict] = {}

        # Configuration
        self.uq_method   = uq_method
        self.model_name  = model_name
        self._query_count = 0

        print(f"\n{'='*55}")
        print(f"  Orchestrator initialized")
        print(f"  Model:      {model_name}")
        print(f"  UQ method:  {uq_method}")
        print(f"  Agents:     {len(self.agents)}")
        print(f"{'='*55}\n")

    # ── Main entry point ──────────────────────

    def run(
        self,
        question:     str,
        choices:      List[str],
        ground_truth: Optional[str] = None,
        task:         str = "unknown",
        verbose:      bool = True,
    ) -> Dict:
        """
        Run the full 5-layer pipeline for a single query.

        Returns a result dict with all blackboard outputs,
        message log, and pipeline events for demo display.
        """
        self._query_count += 1
        total_start = time.time()

        # Reset blackboard and logs for new query
        self.bb.reset()
        self.message_log     = []
        self.pipeline_events = []
        self.blackboard_snapshots = {}

        self._emit_event("PIPELINE_START", {
            "query_id":    self._query_count,
            "question":    question[:80] + "..." if len(question) > 80 else question,
            "choices":     choices,
            "uq_method":   self.uq_method,
        })

        # Write query to blackboard (Layer 0)
        self.bb.write(
            BBKey.QUERY,
            {
                "question":     question,
                "choices":      choices,
                "ground_truth": ground_truth,
                "task":         task,
            },
            author = cfg.AGENT_ORCHESTRATOR,
            layer  = 0,
        )

        if ground_truth:
            self.bb.write(
                BBKey.GROUND_TRUTH, ground_truth,
                author=cfg.AGENT_ORCHESTRATOR, layer=0
            )

        # ── Execute pipeline layers ───────────
        pipeline = [
            ("preprocessing", "Layer 1 — Preprocessing"),
            ("reasoning",     "Layer 2 — Reasoning (HF API)"),
            ("uncertainty",   "Layer 3 — Uncertainty (VIB)"),
            ("calibration",   "Layer 4 — Calibration (ECE)"),
            ("pomdp",         "Layer 5 — POMDP Decision"),
        ]

        for agent_key, layer_label in pipeline:
            agent = self.agents[agent_key]
            layer_num = agent.layer

            if verbose:
                print(f"  → {layer_label}...", end=" ", flush=True)

            msg = make_query_message(
                sender       = cfg.AGENT_ORCHESTRATOR,
                receiver     = agent.agent_id,
                question     = question,
                choices      = choices,
                ground_truth = ground_truth,
                task         = task,
            )

            layer_start = time.time()
            response    = agent.execute(msg)
            layer_time  = time.time() - layer_start

            # Log message exchange
            self._log_message(msg, response, layer_time)
            self._emit_event(f"LAYER_{layer_num}_COMPLETE", {
                "agent":   agent.agent_id,
                "elapsed": layer_time,
                "status":  response.status.value,
            })

            # Snapshot blackboard after each layer
            self.blackboard_snapshots[layer_num] = self.bb.snapshot()

            if response.status == MessageStatus.FAILED:
                if verbose:
                    print(f"FAILED ({response.payload.get('error', 'unknown')})")
                self._emit_event("LAYER_FAILED", {
                    "layer":  layer_num,
                    "agent":  agent.agent_id,
                    "error":  response.payload.get("error", "unknown"),
                })
                # Continue pipeline — partial results still useful
            else:
                if verbose:
                    print(f"done ({layer_time:.2f}s)")

        total_time = time.time() - total_start

        # ── Collect final result ──────────────
        result = self._collect_result(total_time)

        self._emit_event("PIPELINE_COMPLETE", {
            "total_time": total_time,
            "action":     result.get("pomdp_action"),
            "correct":    result.get("is_correct"),
        })

        if verbose:
            self._print_result_summary(result)

        return result

    # ── Batch evaluation ──────────────────────

    def evaluate_batch(
        self,
        dataset:  List[Dict],
        verbose:  bool = True,
    ) -> List[Dict]:
        """Run pipeline on entire dataset. Returns list of results."""
        results   = []
        n_correct = 0

        for i, item in enumerate(dataset):
            result = self.run(
                question     = item["question"],
                choices      = item["choices"],
                ground_truth = item.get("answer"),
                task         = item.get("task", "unknown"),
                verbose      = False,
            )
            results.append(result)

            if result.get("is_correct"):
                n_correct += 1

            if verbose and (i + 1) % 10 == 0:
                acc      = n_correct / (i + 1)
                avg_conf = sum(r.get("confidence", 0) for r in results) / len(results)
                print(
                    f"  [{i+1:4d}/{len(dataset)}] "
                    f"Acc={acc:.3f}  "
                    f"AvgConf={avg_conf:.3f}"
                )

        return results

    # ── Dynamic UQ method switching ───────────

    def set_uq_method(self, method: str):
        """Switch UQ method without rebuilding agents."""
        self.agents["uncertainty"].uq_method = method
        self.uq_method = method

    # ── Internal helpers ──────────────────────

    def _collect_result(self, total_time: float) -> Dict:
        """Read all relevant values from blackboard into result dict."""
        bb = self.bb
        return {
            # Core prediction
            "predicted":        bb.read(BBKey.PREDICTED),
            "predicted_idx":    bb.read(BBKey.PREDICTED_IDX),
            "confidence":       bb.read(BBKey.CONFIDENCE),
            "conf_dist":        bb.read(BBKey.CONF_DIST),
            "log_probs":        bb.read(BBKey.LOG_PROBS),

            # Uncertainty
            "sigma":            bb.read(BBKey.SIGMA),
            "uq_method":        bb.read(BBKey.UQ_METHOD),
            "all_uq_results":   bb.read(BBKey.ALL_UQ_RESULTS),

            # Calibration
            "ece":              bb.read(BBKey.ECE),
            "auroc":            bb.read(BBKey.AUROC),
            "calibration_gap":  bb.read(BBKey.CALIBRATION_GAP),
            "bin_data":         bb.read(BBKey.BIN_DATA),

            # POMDP
            "pomdp_action":     bb.read(BBKey.POMDP_ACTION),
            "pomdp_reason":     bb.read(BBKey.POMDP_REASON),
            "belief_state":     bb.read(BBKey.BELIEF_STATE),

            # Reasoning
            "reasoning_trace":  bb.read(BBKey.REASONING_TRACE),
            "task_type":        bb.read(BBKey.TASK_TYPE),

            # Meta
            "ground_truth":     bb.read(BBKey.GROUND_TRUTH),
            "is_correct":       bb.read(BBKey.IS_CORRECT),
            "errors":           bb.read(BBKey.ERRORS, []),
            "total_time":       total_time,

            # Pipeline artifacts
            "message_log":         self.message_log,
            "pipeline_events":     self.pipeline_events,
            "blackboard_snapshots": self.blackboard_snapshots,
        }

    def _log_message(self, request: Message, response: Message, elapsed: float):
        self.message_log.append({
            "request_id":   request.message_id,
            "response_id":  response.message_id,
            "from":         request.sender,
            "to":           request.receiver,
            "reply_from":   response.sender,
            "status":       response.status.value,
            "elapsed":      elapsed,
            "timestamp":    datetime.utcnow().isoformat(),
            "trace":        response.trace,
        })

    def _emit_event(self, event_type: str, data: Dict):
        self.pipeline_events.append({
            "event":     event_type,
            "data":      data,
            "timestamp": datetime.utcnow().isoformat(),
        })

    def _print_result_summary(self, result: Dict):
        action_icons = {"ANSWER": "✅", "SEEK_INFO": "🔍", "DEFER": "👤"}
        action = result.get("pomdp_action", "—")
        icon   = action_icons.get(action, "?")

        print(f"\n  {'─'*50}")
        print(f"  Predicted:     {result.get('predicted', '—')}")
        print(f"  Confidence:    {result.get('confidence', 0):.4f}")
        print(f"  σ (VIB):       {result.get('sigma', 0):.4f}")
        print(f"  POMDP Action:  {icon} {action}")
        if result.get("is_correct") is not None:
            c = "✓" if result["is_correct"] else "✗"
            print(f"  Correct:       {c}  (GT: {result.get('ground_truth')})")
        if result.get("ece") is not None:
            print(f"  ECE:           {result['ece']:.4f}")
        print(f"  Total time:    {result['total_time']:.2f}s")
        print(f"  {'─'*50}\n")

    def get_agent_statuses(self) -> Dict:
        """Returns status of all agents for demo display."""
        return {
            k: {
                "agent_id":   a.agent_id,
                "role":       a.role.value,
                "layer":      a.layer,
                "call_count": a._call_count,
            }
            for k, a in self.agents.items()
        }


# ── Quick test ────────────────────────────────
if __name__ == "__main__":
    orch = Orchestrator(use_mock=True)

    result = orch.run(
        question = (
            "An insurance applicant is 65 years old, had 4 accidents, "
            "filed 3 claims, smoker: yes, BMI 39.2. "
            "Classify the risk level as high, medium, or low."
        ),
        choices      = ["high", "medium", "low"],
        ground_truth = "high",
        task         = "risk_classification",
        verbose      = True,
    )

    print("\nBlackboard final state:")
    orch.bb.display_layers()

    print("\nMessage log:")
    for msg in result["message_log"]:
        print(f"  {msg['from']:<25} → {msg['to']:<25} [{msg['status']}] {msg['elapsed']:.3f}s")
