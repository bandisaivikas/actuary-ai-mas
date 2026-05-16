# ─────────────────────────────────────────────
# config.py  —  single source of truth
# ─────────────────────────────────────────────

# ── HuggingFace API ──────────────────────────
# All models below work on provider="hf-inference" (HF's own servers).
# We force this provider to avoid featherless-ai / together-ai routing
# which may be down or have different API requirements.
#
# Recommended for this project:
#   "mistralai/Mistral-7B-Instruct-v0.3"   → best reasoning, ~10s
#   "HuggingFaceH4/zephyr-7b-beta"         → good, ~10s
#   "Qwen/Qwen2.5-7B-Instruct"             → fast, reliable
#   "microsoft/Phi-3.5-mini-instruct"      → lightweight, ~5s
HF_MODEL        = "mistralai/Mistral-7B-Instruct-v0.3"
HF_MODEL_SMALL  = "mistralai/Mistral-7B-Instruct-v0.3"  # mistral:latest on Ollama, most reliable local
MAX_NEW_TOKENS  = 150
TEMPERATURE_GEN = 0.1    # generation temperature (low = deterministic)

# ── Dataset ──────────────────────────────────
N_QUESTIONS  = 500
N_TEST       = 100
N_CALIB      = 100
RANDOM_SEED  = 42

# ── Calibration ──────────────────────────────
N_BINS = 10

# ── VIB ──────────────────────────────────────
VIB_RANK     = 32
VIB_BETA     = 0.01

# ── POMDP thresholds ─────────────────────────
SIGMA_ANSWER_THRESHOLD = 0.25   # σ < this → Answer
SIGMA_SEEK_THRESHOLD   = 0.55   # σ < this → Seek Info, else Defer
CONF_MIN_ANSWER        = 0.65   # confidence must also exceed this to Answer

# ── Agent IDs ────────────────────────────────
AGENT_ORCHESTRATOR  = "orchestrator"
AGENT_PREPROCESSING = "preprocessing_agent"
AGENT_REASONING     = "reasoning_agent"
AGENT_UNCERTAINTY   = "uncertainty_agent"
AGENT_CALIBRATION   = "calibration_agent"
AGENT_POMDP         = "pomdp_agent"

# ── Paths ─────────────────────────────────────
DATA_PATH    = "data/actuarial_dataset.json"
RESULTS_PATH = "results/"
