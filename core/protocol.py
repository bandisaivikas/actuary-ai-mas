# ─────────────────────────────────────────────
# core/protocol.py
#
# COMMUNICATION PROTOCOL
#
# Every message in the MAS follows this structure.
# No agent ever passes raw data directly — everything
# is a Message object serialized to JSON.
#
# Message types:
#   REQUEST   — agent asking another agent to do work
#   RESPONSE  — agent returning completed work
#   BROADCAST — agent publishing observation to all
#   ERROR     — agent reporting failure
#   ACK       — agent acknowledging receipt
#   BLACKBOARD_WRITE — agent writing to shared state
#   BLACKBOARD_READ  — agent reading from shared state
# ─────────────────────────────────────────────

import json
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional
from dataclasses import dataclass, field, asdict


class MessageType(str, Enum):
    REQUEST          = "REQUEST"
    RESPONSE         = "RESPONSE"
    BROADCAST        = "BROADCAST"
    ERROR            = "ERROR"
    ACK              = "ACK"
    BLACKBOARD_WRITE = "BLACKBOARD_WRITE"
    BLACKBOARD_READ  = "BLACKBOARD_READ"


class MessageStatus(str, Enum):
    PENDING    = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED  = "COMPLETED"
    FAILED     = "FAILED"


class AgentRole(str, Enum):
    """
    Explicit roles each agent plays in the MAS.
    Used by orchestrator for routing decisions.
    """
    COORDINATOR  = "COORDINATOR"   # Orchestrator
    PREPROCESSOR = "PREPROCESSOR"  # Layer 1
    REASONER     = "REASONER"      # Layer 2 — calls HF API
    UNCERTAINTY  = "UNCERTAINTY"   # Layer 3 — VIB, MC Dropout etc
    CALIBRATOR   = "CALIBRATOR"    # Layer 4 — ECE, AUROC
    DECISION     = "DECISION"      # Layer 5 — POMDP action


@dataclass
class Message:
    """
    The atomic unit of communication in the MAS.

    Every interaction between agents — whether a task request,
    a result, an error, or a blackboard update — is a Message.

    This enforces the communication protocol:
    - Sender and receiver are always explicit
    - Every message has a unique ID for tracing
    - Timestamps enable latency analysis
    - Payload is typed by message_type
    """
    message_type : MessageType
    sender       : str
    receiver     : str          # agent ID or "broadcast" or "blackboard"
    payload      : Dict[str, Any]
    message_id   : str          = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp    : str          = field(default_factory=lambda: datetime.utcnow().isoformat())
    reply_to     : Optional[str] = None    # original message_id this replies to
    status       : MessageStatus = MessageStatus.PENDING
    trace        : list         = field(default_factory=list)  # routing history

    def to_json(self) -> str:
        d = asdict(self)
        d["message_type"] = self.message_type.value
        d["status"]       = self.status.value
        return json.dumps(d, indent=2)

    @classmethod
    def from_json(cls, s: str) -> "Message":
        d    = json.loads(s)
        d["message_type"] = MessageType(d["message_type"])
        d["status"]       = MessageStatus(d["status"])
        return cls(**d)

    def add_trace(self, agent_id: str, note: str = ""):
        """Records routing path for debugging and demo visualization."""
        self.trace.append({
            "agent":     agent_id,
            "timestamp": datetime.utcnow().isoformat(),
            "note":      note,
        })

    def reply(
        self,
        sender: str,
        payload: Dict,
        status: MessageStatus = MessageStatus.COMPLETED,
    ) -> "Message":
        """Creates a RESPONSE message replying to this message."""
        return Message(
            message_type = MessageType.RESPONSE,
            sender       = sender,
            receiver     = self.sender,
            payload      = payload,
            reply_to     = self.message_id,
            status       = status,
        )

    def error(self, sender: str, reason: str) -> "Message":
        """Creates an ERROR message replying to this message."""
        return Message(
            message_type = MessageType.ERROR,
            sender       = sender,
            receiver     = self.sender,
            payload      = {"error": reason, "original_payload": self.payload},
            reply_to     = self.message_id,
            status       = MessageStatus.FAILED,
        )


# ── Message factory helpers ───────────────────

def make_query_message(
    sender:      str,
    receiver:    str,
    question:    str,
    choices:     list,
    ground_truth: Optional[str] = None,
    task:        str = "unknown",
) -> Message:
    """Create a standard actuarial query message."""
    return Message(
        message_type = MessageType.REQUEST,
        sender       = sender,
        receiver     = receiver,
        payload      = {
            "question":     question,
            "choices":      choices,
            "ground_truth": ground_truth,
            "task":         task,
        }
    )


def make_blackboard_write(
    sender: str,
    key:    str,
    value:  Any,
    layer:  int,
) -> Message:
    """Create a blackboard write message."""
    return Message(
        message_type = MessageType.BLACKBOARD_WRITE,
        sender       = sender,
        receiver     = "blackboard",
        payload      = {
            "key":   key,
            "value": value,
            "layer": layer,
        }
    )


# ── Quick test ────────────────────────────────
if __name__ == "__main__":
    msg = make_query_message(
        sender       = "orchestrator",
        receiver     = "reasoning_agent",
        question     = "Is this claim suspicious?",
        choices      = ["suspicious", "not suspicious"],
        ground_truth = "suspicious",
        task         = "fraud_detection",
    )
    print(msg.to_json())
    print(f"\nMessage ID:  {msg.message_id}")
    print(f"Type:        {msg.message_type}")

    # Test reply
    reply = msg.reply(
        sender  = "reasoning_agent",
        payload = {"predicted": "suspicious", "confidence": 0.82},
    )
    print(f"\nReply to: {reply.reply_to}")
    print(f"From:     {reply.sender}")
