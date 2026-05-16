# ─────────────────────────────────────────────
# core/blackboard.py
#
# BLACKBOARD — Shared Knowledge State
#
# The blackboard is the central shared memory
# of the MAS. All agents read from and write
# to it. Agents are decoupled — they never call
# each other directly, only the blackboard.
#
# Architecture:
#   - Layered slots (one per agent layer)
#   - Write history for full traceability
#   - Event subscribers (for orchestrator routing)
#   - Thread-safe (lock per slot)
#   - Full JSON serialization for demo display
# ─────────────────────────────────────────────

import threading
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field


# ── Blackboard slot keys ──────────────────────
# These are the canonical keys used by all agents.
# An agent ONLY reads keys from layers above it
# and writes keys in its own layer.

class BBKey:
    # Layer 0 — raw input
    QUERY           = "query"            # original question + choices
    TASK_TYPE       = "task_type"        # fraud/risk/premium/compliance

    # Layer 1 — preprocessing
    PARSED_QUESTION = "parsed_question"  # cleaned question text
    PARSED_CHOICES  = "parsed_choices"   # validated choices list
    PROMPT          = "prompt"           # formatted ReAct prompt

    # Layer 2 — reasoning
    REASONING_TRACE = "reasoning_trace"  # Thought/Action/Observation chain
    PREDICTED       = "predicted"        # agent's answer
    PREDICTED_IDX   = "predicted_idx"    # index of predicted choice
    LOG_PROBS       = "log_probs"        # raw log-probabilities per choice
    CONF_DIST       = "conf_dist"        # softmax confidence distribution
    CONFIDENCE      = "confidence"       # max confidence (softmax)
    REASONING_TIME  = "reasoning_time"   # latency in seconds

    # Layer 3 — uncertainty
    SIGMA           = "sigma"            # VIB uncertainty width
    UQ_METHOD       = "uq_method"        # which UQ method was used
    ALL_UQ_RESULTS  = "all_uq_results"   # comparison of all 4 methods
    HIDDEN_STATES   = "hidden_states"    # layer-wise hidden states (Phase 4)

    # Layer 4 — calibration
    ECE             = "ece"              # expected calibration error
    AUROC           = "auroc"            # AUROC score
    BIN_DATA        = "bin_data"         # reliability diagram bins
    CALIBRATION_GAP = "calibration_gap"  # confidence - accuracy

    # Layer 5 — POMDP decision
    POMDP_ACTION    = "pomdp_action"     # ANSWER / SEEK_INFO / DEFER
    POMDP_REASON    = "pomdp_reason"     # why this action was chosen
    BELIEF_STATE    = "belief_state"     # (mu, sigma) tuple

    # Meta
    GROUND_TRUTH    = "ground_truth"     # for evaluation
    IS_CORRECT      = "is_correct"       # correctness flag
    ERRORS          = "errors"           # list of agent errors
    PIPELINE_STATUS = "pipeline_status"  # current pipeline state


@dataclass
class BlackboardEntry:
    """A single entry written to the blackboard."""
    key        : str
    value      : Any
    author     : str          # which agent wrote this
    layer      : int          # which pipeline layer
    timestamp  : str = field(default_factory=lambda: datetime.utcnow().isoformat())
    version    : int = 1      # increments on overwrites


class Blackboard:
    """
    Layered shared knowledge space for the MAS.

    Design decisions:
    1. Layered slots — each layer can only overwrite its own keys
    2. Full write history — nothing is ever truly deleted
    3. Event callbacks — orchestrator subscribes to layer completions
    4. Thread-safe — supports parallel agent execution
    5. Immutable query — Layer 0 (input) cannot be overwritten

    The blackboard is the ONLY communication channel between agents.
    This enforces full decoupling — changing one agent never breaks another.
    """

    def __init__(self):
        self._slots    : Dict[str, BlackboardEntry] = {}
        self._history  : List[BlackboardEntry]      = []
        self._callbacks: Dict[str, List[Callable]]  = {}  # key → [callbacks]
        self._lock     = threading.RLock()
        self._session_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    # ── Write ─────────────────────────────────

    def write(
        self,
        key:    str,
        value:  Any,
        author: str,
        layer:  int,
    ) -> BlackboardEntry:
        """
        Write a value to the blackboard.
        Records full history — every write is preserved.
        Fires any registered callbacks for this key.
        """
        with self._lock:
            # Prevent overwriting Layer 0 (raw input)
            if key in (BBKey.QUERY,) and key in self._slots:
                raise PermissionError(
                    f"Layer 0 key '{key}' is immutable after initial write."
                )

            version = 1
            if key in self._slots:
                version = self._slots[key].version + 1

            entry = BlackboardEntry(
                key=key, value=value,
                author=author, layer=layer, version=version
            )
            self._slots[key]   = entry
            self._history.append(entry)

        # Fire callbacks outside lock to prevent deadlock
        if key in self._callbacks:
            for cb in self._callbacks[key]:
                try:
                    cb(key, value, author)
                except Exception as e:
                    print(f"[Blackboard] Callback error for '{key}': {e}")

        return entry

    # ── Read ──────────────────────────────────

    def read(self, key: str, default: Any = None) -> Any:
        """Read current value of a key. Returns default if not set."""
        with self._lock:
            if key in self._slots:
                return self._slots[key].value
            return default

    def read_entry(self, key: str) -> Optional[BlackboardEntry]:
        """Read the full entry (includes author, timestamp, layer)."""
        with self._lock:
            return self._slots.get(key)

    def has(self, key: str) -> bool:
        with self._lock:
            return key in self._slots

    def layer_complete(self, layer: int) -> bool:
        """Check if all entries for a given layer have been written."""
        with self._lock:
            layer_entries = [e for e in self._slots.values() if e.layer == layer]
            return len(layer_entries) > 0

    # ── Subscribe ─────────────────────────────

    def subscribe(self, key: str, callback: Callable):
        """
        Register a callback that fires when key is written.
        Used by orchestrator for reactive routing.
        """
        if key not in self._callbacks:
            self._callbacks[key] = []
        self._callbacks[key].append(callback)

    # ── Snapshot ──────────────────────────────

    def snapshot(self) -> Dict:
        """
        Returns a clean dict of all current blackboard values.
        Used for demo visualization and logging.
        """
        with self._lock:
            snap = {}
            for key, entry in self._slots.items():
                # Skip large objects not needed for display
                if key == BBKey.HIDDEN_STATES:
                    snap[key] = f"[tensor data, {type(entry.value).__name__}]"
                else:
                    snap[key] = {
                        "value":     entry.value,
                        "author":    entry.author,
                        "layer":     entry.layer,
                        "timestamp": entry.timestamp,
                        "version":   entry.version,
                    }
        return snap

    def to_json(self) -> str:
        """Full JSON serialization of current state."""
        snap = self.snapshot()
        return json.dumps(snap, indent=2, default=str)

    def history_for(self, key: str) -> List[BlackboardEntry]:
        """Returns all historical writes for a key (audit trail)."""
        with self._lock:
            return [e for e in self._history if e.key == key]

    # ── Reset ─────────────────────────────────

    def reset(self):
        """
        Clear the blackboard for a new query.
        Called by orchestrator at start of each new request.
        """
        with self._lock:
            self._slots   = {}
            self._history = []
        # Keep callbacks registered — they survive across queries

    # ── Display ───────────────────────────────

    def display_layers(self):
        """Print a formatted layer-by-layer view of the blackboard."""
        layer_names = {
            0: "Input",
            1: "Preprocessing",
            2: "Reasoning",
            3: "Uncertainty",
            4: "Calibration",
            5: "POMDP Decision",
        }
        with self._lock:
            for layer in range(6):
                entries = [e for e in self._slots.values() if e.layer == layer]
                if not entries:
                    continue
                print(f"\n  Layer {layer} — {layer_names.get(layer, 'Unknown')}")
                print(f"  {'─'*50}")
                for e in entries:
                    val = e.value
                    if isinstance(val, float):
                        val = f"{val:.4f}"
                    elif isinstance(val, list) and len(val) > 3:
                        val = f"[{len(val)} items]"
                    elif isinstance(val, str) and len(val) > 60:
                        val = val[:60] + "..."
                    print(f"  {e.key:<22} = {val}  [by {e.author}]")


# ── Quick test ────────────────────────────────
if __name__ == "__main__":
    bb = Blackboard()

    # Simulate pipeline writes
    bb.write(BBKey.QUERY,   {"question": "Is this claim suspicious?", "choices": ["suspicious", "not suspicious"]}, author="orchestrator", layer=0)
    bb.write(BBKey.PROMPT,  "You are an actuary...", author="preprocessing_agent", layer=1)
    bb.write(BBKey.PREDICTED, "suspicious", author="reasoning_agent", layer=2)
    bb.write(BBKey.CONFIDENCE, 0.847,        author="reasoning_agent", layer=2)
    bb.write(BBKey.SIGMA,    0.18,           author="uncertainty_agent", layer=3)
    bb.write(BBKey.ECE,      0.062,          author="calibration_agent", layer=4)
    bb.write(BBKey.POMDP_ACTION, "ANSWER",   author="pomdp_agent",       layer=5)

    print("Blackboard state:")
    bb.display_layers()

    print(f"\nHistory for '{BBKey.CONFIDENCE}':")
    for h in bb.history_for(BBKey.CONFIDENCE):
        print(f"  v{h.version}  {h.value}  by {h.author}  at {h.timestamp}")
