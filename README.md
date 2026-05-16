# Actuary AI — Multi-Agent Uncertainty System
### Project 9 | Agentic AI | IIIT Naya Raipur

> **Research Question:** Does IB-selective quantization make LLM agents not just wrong — but *confidently* wrong? Can a VIB layer fix it?

**Answer from Phase 3 (100 questions, Mistral 7B):**
- Softmax ECE: **0.2945** — model said 84.5% confident, was right 68% of the time
- VIB Layer ECE: **0.1501** — **49% ECE reduction** through information-theoretic correction

---

## Demo

```bash
# 1 — Install Ollama (free local inference)
brew install ollama && ollama pull mistral

# 2 — Install dependencies
pip install streamlit plotly torch numpy pandas huggingface_hub

# 3 — Generate dataset
PYTHONPATH=. python3 data/dataset_generator.py

# 4 — Launch demo
PYTHONPATH=. streamlit run demo_app.py
```

Opens at **http://localhost:8501**

No Ollama? The system auto-detects and falls back to Mock mode.

---

## Architecture

**Layered Blackboard + Orchestrator MAS**

```
Query
  ↓
Orchestrator (Layer 0) ── controls pipeline, routes messages
  ↓
[Blackboard — shared state, all agents read/write here]
  ↓
Layer 1: PreprocessingAgent  → parses question, builds prompt
Layer 2: ReasoningAgent      → HF API/Ollama inference, confidence
Layer 3: UncertaintyAgent    → 4 UQ methods, VIB σ
Layer 4: CalibrationAgent    → ECE, AUROC, calibration gap
Layer 5: PomdpAgent          → ANSWER / SEEK_INFO / DEFER
```

**Communication:** Every agent interaction is a structured JSON message with `type`, `sender`, `receiver`, `payload`, `timestamp`, and routing `trace`. Agents never call each other directly — only through the blackboard.

---

## Project Structure

```
├── config.py                    # All hyperparameters
├── demo_app.py                  # Streamlit live demo
├── phase3_evaluate.py           # UQ comparison on 100 questions
├── phase4_quantization.py       # Quantization ablation FP16→INT8→INT4
├── phase5_vib_train.py          # Train VIB encoder
├── prewarm_demo.py              # Pre-seed queries before demo
├── run_ground_truth_queries.py  # Ground truth evaluation
│
├── core/
│   ├── protocol.py              # Message types + communication protocol
│   ├── blackboard.py            # Shared state bus
│   ├── base_agent.py            # Base class all agents inherit
│   ├── hf_api.py                # Inference backend (Ollama/Groq/HF/Mock)
│   └── orchestrator.py          # Pipeline controller
│
├── agents/
│   └── agents.py                # All 5 agents
│
├── data/
│   └── dataset_generator.py     # 500 synthetic actuarial questions
│
├── notebooks/
│   └── phase4_quantization_lab.ipynb   # GPU quantization (A100)
│
└── results/
    ├── phase3/                  # UQ comparison results + plots
    ├── phase4/                  # Quantization ablation results
    └── phase5/                  # Trained VIB encoder
```

---

## Inference Backends

The system auto-detects the best available backend:

| Backend | Speed | Cost | Setup |
|---|---|---|---|
| 🟢 **Ollama** (recommended) | 2–5s | Free | `brew install ollama && ollama pull mistral` |
| 🔵 **Groq** | 1–2s | Free | `export GROQ_API_KEY=gsk_...` |
| 🔴 **HuggingFace API** | 10–15s | Credits | `export HF_API_KEY=hf_...` |
| ⚪ **Mock** | Instant | Free | Automatic fallback |

---

## The 4 UQ Methods

| Method | How it works | Phase 3 ECE |
|---|---|---|
| Softmax | Max softmax probability | 0.2945 |
| Temperature Scaling | Divide logits by T=1.5 | 0.2348 |
| MC Dropout | N=5 stochastic passes, variance | 0.2935 |
| **VIB Layer** ★ | Latent Gaussian σ from trained encoder | **0.1501** |

★ Novel contribution — 49% ECE reduction over baseline

---

## Running the Evaluation Phases

### Phase 3 — UQ Method Comparison (~10 min with Ollama)
```bash
PYTHONPATH=. python3 phase3_evaluate.py
# → results/phase3/  (ECE table, 4 plots)
```

### Phase 4 — Quantization Ablation (mock: instant, real: A100)
```bash
PYTHONPATH=. python3 phase4_quantization.py --mock
# → results/phase4/  (ECE by precision, layer heatmap)
```

### Phase 5 — Train VIB Encoder (~2 min)
```bash
PYTHONPATH=. python3 phase5_vib_train.py
# → results/phase5/vib_encoder.pt
```

### Pre-warm before demo
```bash
PYTHONPATH=. python3 prewarm_demo.py
# Seeds 10 queries → rolling ECE active from demo start
```

---

## Key Results

### Phase 3 — Real evaluation (Mistral 7B, 100 questions)

```
Method               ECE      CalGap   AUROC
─────────────────────────────────────────────
Softmax             0.2945   +0.1656   0.45    ← severely overconfident
Temperature Scale   0.2348   +0.0886   0.45
MC Dropout          0.2935   +0.1634   0.48
VIB Layer ★         0.1501   -0.0805   0.37    ← best calibration
```

### Phase 4 — Quantization (simulated, consistent with literature)

```
FP16  ECE=0.052   ← baseline
INT8  ECE=0.114   (+119%)
INT4  ECE=0.273   (+425%) ← degradation in IB-bottleneck layers
INT4+VIB ECE=0.081 ← VIB recovers calibration
```

---

## Research Context

This is **Stage 3 of Research Thread 3 — Information-Aware Agentic LLM Training**:

| Stage | Status | Description |
|-------|--------|-------------|
| 1 | ✅ Done | IB theory + Information Plane on GPT-2 |
| 2 | ✅ Done | Silent Inference Degradation (72% KL, 31× displacement) |
| **3** | **🔵 This project** | **Miscalibration + VIB Layer fix** |
| 4 | 🔶 Next | POMDP + SAC Deep RL agent |
| 5 | ⬜ Future | CALM curiosity-driven exploration |

**Paper title:**
> *"Quantization-Induced Miscalibration in LLM Agents: An IB Analysis and VIB-Based Correction"*

---

## Author

**Bandisaivikas (Vicky)**
B.Tech Data Science & AI, IIIT Naya Raipur, Batch 2027
GitHub: [@bandisaivikas](https://github.com/bandisaivikas)
