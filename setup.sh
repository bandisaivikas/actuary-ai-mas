#!/bin/bash
# ─────────────────────────────────────────────
# setup.sh — One-command project setup
# Usage: bash setup.sh
# ─────────────────────────────────────────────

echo ""
echo "═══════════════════════════════════════════"
echo "  Actuary AI — MAS Setup"
echo "═══════════════════════════════════════════"

# 1. Python dependencies
echo ""
echo "→ Installing Python dependencies..."
pip install -r requirements.txt -q

# 2. Generate dataset
echo "→ Generating actuarial dataset..."
PYTHONPATH=. python3 data/dataset_generator.py

# 3. Run Phase 4 mock (quantization results for demo)
echo "→ Running Phase 4 mock (quantization chart)..."
PYTHONPATH=. python3 phase4_quantization.py --mock

# 4. Check Ollama
echo ""
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✓ Ollama detected — running locally"
    echo "  Available models:"
    curl -s http://localhost:11434/api/tags | python3 -c \
        "import sys,json; [print('    -', m['name']) for m in json.load(sys.stdin).get('models',[])]"
else
    echo "⚠ Ollama not running."
    echo "  For best experience: brew install ollama && ollama pull mistral"
    echo "  The demo will use Mock mode without Ollama."
fi

echo ""
echo "═══════════════════════════════════════════"
echo "  Setup complete. Run the demo:"
echo ""
echo "    PYTHONPATH=. streamlit run demo_app.py"
echo ""
echo "  For full evaluation:"
echo "    PYTHONPATH=. python3 phase3_evaluate.py"
echo "    PYTHONPATH=. python3 phase5_vib_train.py"
echo "═══════════════════════════════════════════"
echo ""
