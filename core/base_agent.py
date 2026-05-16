# ─────────────────────────────────────────────
# core/base_agent.py
#
# BASE AGENT
#
# All agents inherit from this class.
# Enforces:
#   - Role declaration
#   - Blackboard-only communication
#   - Structured message handling
#   - Error handling and reporting
#   - Execution logging
# ─────────────────────────────────────────────

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from core.protocol  import Message, MessageType, MessageStatus, AgentRole
from core.blackboard import Blackboard, BBKey


class BaseAgent(ABC):
    """
    Base class for all MAS agents.

    Every agent has:
        - A unique ID
        - A declared role (from AgentRole enum)
        - A layer number (1–5 in the pipeline)
        - A reference to the shared blackboard
        - A structured execute() method

    Agents communicate ONLY through:
        1. Reading from the blackboard
        2. Writing to the blackboard
        3. Sending/receiving Message objects

    Agents NEVER:
        - Call other agents directly
        - Hold references to other agents
        - Maintain state between queries (stateless per query)
    """

    def __init__(
        self,
        agent_id:   str,
        role:       AgentRole,
        layer:      int,
        blackboard: Blackboard,
    ):
        self.agent_id   = agent_id
        self.role       = role
        self.layer      = layer
        self.bb         = blackboard
        self._log       = []      # execution log for this agent
        self._call_count = 0

    # ── Abstract interface ────────────────────

    @abstractmethod
    def process(self, message: Message) -> Message:
        """
        Core logic of the agent.
        Must read inputs from blackboard, execute task,
        write outputs to blackboard, return response message.
        """
        pass

    @property
    @abstractmethod
    def required_keys(self) -> list:
        """
        Blackboard keys this agent needs as input.
        Used by orchestrator to verify layer is ready.
        """
        pass

    @property
    @abstractmethod
    def output_keys(self) -> list:
        """
        Blackboard keys this agent will write.
        Used for documentation and dependency checking.
        """
        pass

    # ── Public interface ──────────────────────

    def execute(self, message: Message) -> Message:
        """
        Wrapper around process() that adds:
        - Input validation (required blackboard keys present)
        - Timing
        - Error handling
        - Logging
        - Trace recording
        """
        self._call_count += 1
        start = time.time()

        message.add_trace(self.agent_id, f"received by {self.role.value}")
        message.status = MessageStatus.PROCESSING

        # Check required blackboard keys are present
        missing = [k for k in self.required_keys if not self.bb.has(k)]
        if missing:
            err_msg = f"Missing required blackboard keys: {missing}"
            self._log_event("ERROR", err_msg)
            return message.error(self.agent_id, err_msg)

        try:
            response = self.process(message)
            elapsed  = time.time() - start

            response.add_trace(
                self.agent_id,
                f"completed in {elapsed:.3f}s"
            )
            self._log_event("SUCCESS", f"processed in {elapsed:.3f}s")
            return response

        except Exception as e:
            elapsed  = time.time() - start
            err_msg  = f"{type(e).__name__}: {str(e)}"
            self._log_event("ERROR", err_msg)

            # Write error to blackboard for orchestrator
            errors = self.bb.read(BBKey.ERRORS, default=[])
            errors.append({
                "agent":   self.agent_id,
                "error":   err_msg,
                "layer":   self.layer,
            })
            self.bb.write(BBKey.ERRORS, errors, self.agent_id, self.layer)

            return message.error(self.agent_id, err_msg)

    # ── Blackboard helpers ────────────────────

    def bb_read(self, key: str, default: Any = None) -> Any:
        """Read from blackboard with agent attribution logging."""
        value = self.bb.read(key, default)
        self._log_event("READ", f"bb[{key}]")
        return value

    def bb_write(self, key: str, value: Any):
        """Write to blackboard with automatic layer attribution."""
        self.bb.write(key, value, author=self.agent_id, layer=self.layer)
        self._log_event("WRITE", f"bb[{key}] = {str(value)[:50]}")

    # ── Logging ───────────────────────────────

    def _log_event(self, event_type: str, detail: str):
        self._log.append({
            "agent":      self.agent_id,
            "event":      event_type,
            "detail":     detail,
            "timestamp":  time.time(),
        })

    def get_log(self) -> list:
        return self._log.copy()

    def clear_log(self):
        self._log = []

    def status_line(self) -> str:
        return (
            f"[{self.agent_id}] "
            f"role={self.role.value} "
            f"layer={self.layer} "
            f"calls={self._call_count}"
        )

    def __repr__(self):
        return f"Agent({self.agent_id}, {self.role.value}, layer={self.layer})"
