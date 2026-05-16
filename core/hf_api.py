# ─────────────────────────────────────────────
# core/hf_api.py  —  v3 multi-backend inference
#
# Backend priority (auto-detected):
#   1. Ollama  — local, free, fastest on M2 Mac
#   2. Groq    — free cloud, very fast
#   3. HF API  — when credits available
#   4. Mock    — always works, for testing
#
# Why Ollama is best for this project:
#   - Runs on M2 MacBook, completely free
#   - No rate limits, no credits, no timeouts
#   - 2-5s per query vs 12-15s via HF API
#   - Full control over the model
#
# SETUP (one time):
#   brew install ollama
#   ollama pull mistral          # 4.1GB, best quality
#   ollama pull llama3.2         # 2.0GB, fastest
#   ollama serve                 # runs in background
#
# Groq (free alternative if no Ollama):
#   Get free key: console.groq.com
#   export GROQ_API_KEY=gsk_...
#   pip install groq
# ─────────────────────────────────────────────

import os, time, math, json
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple


# ── Backend detection ─────────────────────────

def detect_backend() -> str:
    """
    Auto-detect which inference backend is available.
    Returns: 'ollama' | 'groq' | 'hf' | 'mock'
    """
    # Check Ollama (local)
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        return "ollama"
    except Exception:
        pass

    # Check Groq API key
    if os.getenv("GROQ_API_KEY"):
        return "groq"

    # Check HF API key
    if os.getenv("HF_API_KEY"):
        return "hf"

    return "mock"


# ── Ollama client (local, free) ───────────────

class OllamaClient:
    """
    Ollama local inference client.
    Fastest option — 2-5s per query on M2 MacBook.
    No API key, no credits, no rate limits.

    Setup:
        brew install ollama
        ollama pull mistral        # recommended
        ollama serve               # start server
    """

    def __init__(self, model_name: str = "mistral"):
        # Normalize model name for Ollama
        # HF names like "mistralai/Mistral-7B-Instruct-v0.3" → "mistral"
        self.model_name = self._normalize_model(model_name)
        self.base_url   = "http://localhost:11434"
        self._verify_model()

    def _normalize_model(self, name: str) -> str:
        """Map HF model names to Ollama model tags."""
        mapping = {
            "mistralai/Mistral-7B-Instruct-v0.3": "mistral",
            "mistralai/Mistral-7B-Instruct-v0.1": "mistral",
            "HuggingFaceH4/zephyr-7b-beta":        "mistral",
            "meta-llama/Llama-3.2-3B-Instruct":    "llama3.2",
            "meta-llama/Llama-3.1-8B-Instruct":    "llama3.1",
            "Qwen/Qwen2.5-7B-Instruct":            "qwen2.5",
            "Qwen/Qwen2.5-14B-Instruct":           "qwen2.5:14b",
            "microsoft/Phi-3.5-mini-instruct":      "phi3",
            "microsoft/Phi-3-mini-4k-instruct":     "phi3:mini",
        }
        return mapping.get(name, name.split("/")[-1].lower().split("-instruct")[0])

    def _verify_model(self):
        """
        Check model is available on Ollama.
        Resolves to the exact name the API accepts.

        Key insight: Ollama's /api/generate accepts EITHER:
          - bare name "mistral" (resolves to mistral:latest)
          - full tag "qwen2.5:14b"
        We use bare name when possible to keep it simple.
        """
        try:
            import urllib.request
            resp  = urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=3)
            tags  = json.loads(resp.read())
            models = tags.get("models", [])

            available_full = [m["name"] for m in models]
            available_bare = [m["name"].split(":")[0] for m in models]
            requested      = self.model_name

            # 1. Exact full tag match — use as-is
            if requested in available_full:
                pass

            # 2. Bare name match — use the full tag to be safe
            #    "qwen2.5" → "qwen2.5:14b"  (Ollama may not resolve bare names)
            #    "mistral" → "mistral:latest" is fine, stays as "mistral"
            elif requested in available_bare:
                for m in models:
                    if m["name"].split(":")[0] == requested:
                        full_tag = m["name"]
                        # Only use full tag if it's not just "name:latest"
                        # (bare "mistral" works, but "qwen2.5" needs ":14b")
                        if "latest" not in full_tag:
                            self.model_name = full_tag
                            print(f"  Model resolved: '{requested}' → '{self.model_name}'")
                        break

            # 3. Prefix match — use the full tag
            #    e.g. "qwen2.5" → "qwen2.5:14b"
            elif any(f.startswith(requested) or requested in f
                     for f in available_full):
                for full in available_full:
                    if full.startswith(requested) or requested in full:
                        self.model_name = full
                        print(f"  Model resolved: '{requested}' → '{self.model_name}'")
                        break

            # 4. Not found — auto-select best available
            else:
                print(
                    f"\n⚠ Ollama model '{requested}' not found.\n"
                    f"  Available: {available_full}\n"
                )
                preferred = ["mistral", "phi3", "llama3", "qwen"]
                chosen = None
                for pref in preferred:
                    for bare in available_bare:
                        if pref in bare:
                            # find the full name for this bare
                            for full in available_full:
                                if full.split(":")[0] == bare:
                                    chosen = bare  # use bare so Ollama resolves :latest
                                    break
                            if chosen: break
                    if chosen: break
                if not chosen and available_bare:
                    chosen = available_bare[0]
                if chosen:
                    self.model_name = chosen
                    print(f"  Auto-selected: {self.model_name}")

        except Exception as e:
            print(f"  Ollama check: {e}")

        print(f"Ollama client → {self.model_name} (local)")

    def _chat(self, prompt: str, max_tokens: int = 20) -> str:
        """Send a chat request to Ollama and return generated text."""
        import urllib.request, json
        data = json.dumps({
            "model":  self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict":  max_tokens,
                "temperature":  0.01,
                "top_k":        10,
            }
        }).encode()

        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data    = data,
            headers = {"Content-Type": "application/json"},
            method  = "POST",
        )
        resp = urllib.request.urlopen(req, timeout=60)
        result = json.loads(resp.read())
        return result.get("response", "").strip()

    def score_choices(
        self,
        prompt:  str,
        choices: List[str],
    ) -> Tuple[str, float, List[float], List[float]]:
        """
        Score choices by generating a short answer and matching.
        Ollama doesn't expose log-probs via the generate API
        so we use text matching with confidence estimation.
        """
        try:
            generated = self._chat(prompt, max_tokens=15).lower().strip()

            scores = []
            for choice in choices:
                c = choice.lower().strip()
                if c == generated or generated.startswith(c):
                    scores.append(-0.1)   # exact / prefix match
                elif c in generated:
                    scores.append(-0.5)   # contained match
                elif c[:4] in generated:
                    scores.append(-1.2)   # partial match
                elif generated and c[0] == generated[0]:
                    scores.append(-2.0)   # first char match
                else:
                    scores.append(-3.5)   # no match

            # Break ties
            if len(set(scores)) == 1:
                scores[0] += 1.0

            lp_t      = torch.tensor(scores, dtype=torch.float32)
            conf_dist = F.softmax(lp_t, dim=0).tolist()
            pred_idx  = int(torch.argmax(lp_t).item())
            return choices[pred_idx], conf_dist[pred_idx], conf_dist, scores

        except Exception as e:
            print(f"  Ollama scoring failed: {e}")
            return choices[0], 1/len(choices), [1/len(choices)]*len(choices), [-1.0]*len(choices)

    def score_choice(self, prompt: str, choice: str) -> float:
        choices = self._extract_choices(prompt)
        if not choices:
            return -2.0
        _, _, dist, _ = self.score_choices(prompt, choices)
        for i, c in enumerate(choices):
            if c.lower() == choice.lower():
                return math.log(max(dist[i], 1e-10))
        return -2.0

    def _extract_choices(self, prompt: str) -> List[str]:
        for line in prompt.split('\n'):
            if 'options:' in line.lower():
                part = line.split(':', 1)[-1]
                return [c.strip() for c in part.split('/') if c.strip()]
        return []

    def generate_reasoning(
        self, question: str, choices: List[str], system_prompt: str = ""
    ) -> str:
        opts = " / ".join(choices)
        prompt = (
            f"{system_prompt}\n\n"
            f"Question: {question}\n"
            f"Think step by step then answer ({opts}).\n\nThought:"
        )
        try:
            text = self._chat(prompt, max_tokens=120)
            return f"Thought:{text}"
        except Exception as e:
            return f"Thought: [unavailable: {e}]"

    @staticmethod
    def build_scoring_prompt(question: str, choices: List[str], few_shot: str = "") -> str:
        opts = " / ".join(choices)
        return (
            f"You are an expert insurance actuary. "
            f"Pick ONE answer from the options.\n\n"
            f"{few_shot}"
            f"Question: {question}\n"
            f"Options: {opts}\n"
            f"Answer: "
        )


# ── Groq client (free cloud) ──────────────────

class GroqClient:
    """
    Groq API — free tier, very fast (~1-2s).
    Free models: llama-3.1-8b-instant, llama3-8b-8192, gemma2-9b-it
    Get key: console.groq.com (free)
    """

    GROQ_MODELS = {
        "mistralai/Mistral-7B-Instruct-v0.3": "llama-3.1-8b-instant",
        "HuggingFaceH4/zephyr-7b-beta":        "llama-3.1-8b-instant",
        "meta-llama/Llama-3.2-3B-Instruct":    "llama-3.1-8b-instant",
        "Qwen/Qwen2.5-7B-Instruct":            "gemma2-9b-it",
    }
    DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"

    def __init__(self, model_name: str = "llama-3.1-8b-instant"):
        from groq import Groq
        self.groq_model = self.GROQ_MODELS.get(model_name, self.DEFAULT_GROQ_MODEL)
        self.client     = Groq(api_key=os.getenv("GROQ_API_KEY"))
        print(f"Groq client → {self.groq_model} (free cloud)")

    def _chat(self, prompt: str, max_tokens: int = 20) -> str:
        resp = self.client.chat.completions.create(
            model       = self.groq_model,
            messages    = [{"role": "user", "content": prompt}],
            max_tokens  = max_tokens,
            temperature = 0.01,
        )
        return resp.choices[0].message.content.strip()

    def score_choices(
        self, prompt: str, choices: List[str]
    ) -> Tuple[str, float, List[float], List[float]]:
        try:
            generated = self._chat(prompt, max_tokens=15).lower().strip()
            scores = []
            for choice in choices:
                c = choice.lower().strip()
                if c == generated or generated.startswith(c):
                    scores.append(-0.1)
                elif c in generated:
                    scores.append(-0.5)
                elif c[:4] in generated:
                    scores.append(-1.2)
                else:
                    scores.append(-3.5)
            if len(set(scores)) == 1:
                scores[0] += 1.0
            lp_t      = torch.tensor(scores, dtype=torch.float32)
            conf_dist = F.softmax(lp_t, dim=0).tolist()
            pred_idx  = int(torch.argmax(lp_t).item())
            return choices[pred_idx], conf_dist[pred_idx], conf_dist, scores
        except Exception as e:
            print(f"  Groq failed: {e}")
            return choices[0], 1/len(choices), [1/len(choices)]*len(choices), [-1.0]*len(choices)

    def score_choice(self, prompt: str, choice: str) -> float:
        choices = self._extract_choices(prompt)
        if not choices:
            return -2.0
        _, _, dist, _ = self.score_choices(prompt, choices)
        for i, c in enumerate(choices):
            if c.lower() == choice.lower():
                return math.log(max(dist[i], 1e-10))
        return -2.0

    def _extract_choices(self, prompt: str) -> List[str]:
        for line in prompt.split('\n'):
            if 'options:' in line.lower():
                part = line.split(':', 1)[-1]
                return [c.strip() for c in part.split('/') if c.strip()]
        return []

    def generate_reasoning(
        self, question: str, choices: List[str], system_prompt: str = ""
    ) -> str:
        opts   = " / ".join(choices)
        prompt = (
            f"{system_prompt}\n\nQuestion: {question}\n"
            f"Think step by step then answer ({opts}).\n\nThought:"
        )
        try:
            return f"Thought:{self._chat(prompt, max_tokens=120)}"
        except Exception as e:
            return f"Thought: [unavailable: {e}]"

    @staticmethod
    def build_scoring_prompt(question: str, choices: List[str], few_shot: str = "") -> str:
        return OllamaClient.build_scoring_prompt(question, choices, few_shot)


# ── HF API client (when credits available) ────

class HFApiClient:
    """
    HuggingFace Inference API.
    Use when you have credits or a PRO account.
    """

    def __init__(
        self,
        api_key:    Optional[str] = None,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
        max_retries: int = 3,
        retry_delay: float = 3.0,
    ):
        from huggingface_hub import InferenceClient
        self.api_key     = api_key or os.getenv("HF_API_KEY", "")
        self.model_name  = model_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        if not self.api_key:
            raise ValueError("HuggingFace API key required.")

        self.client = InferenceClient(
            model    = self.model_name,
            token    = self.api_key,
            provider = "hf-inference",   # avoid featherless-ai / together-ai
        )
        print(f"HF API client → {self.model_name} [hf-inference]")

    def _generate(self, prompt: str, max_new_tokens: int = 15) -> str:
        for attempt in range(self.max_retries):
            try:
                # Try chat_completion first (Llama, Mistral instruct)
                resp = self.client.chat_completion(
                    messages   = [{"role": "user", "content": prompt}],
                    max_tokens = max_new_tokens,
                    temperature = 0.01,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e1:
                try:
                    # Fallback: text_generation (Zephyr style)
                    resp = self.client.text_generation(
                        prompt,
                        max_new_tokens   = max_new_tokens,
                        return_full_text = False,
                        temperature      = 0.01,
                        do_sample        = False,
                    )
                    return resp.strip() if isinstance(resp, str) else ""
                except Exception as e2:
                    err = str(e2).lower()
                    if "402" in err or "credit" in err or "payment" in err:
                        raise RuntimeError(
                            "HF credits exhausted. Use Ollama or Groq instead.\n"
                            "  ollama pull mistral && ollama serve\n"
                            "  OR: export GROQ_API_KEY=gsk_..."
                        )
                    if "rate limit" in err or "429" in err:
                        time.sleep(self.retry_delay * (attempt + 1))
                    elif attempt == self.max_retries - 1:
                        raise RuntimeError(f"HF failed: {e1} / {e2}")
                    else:
                        time.sleep(self.retry_delay)
        raise RuntimeError("Max retries exceeded")

    def score_choices(
        self, prompt: str, choices: List[str]
    ) -> Tuple[str, float, List[float], List[float]]:
        try:
            generated = self._generate(prompt, max_new_tokens=15).lower().strip()
            scores = []
            for choice in choices:
                c = choice.lower().strip()
                if c == generated or generated.startswith(c): scores.append(-0.1)
                elif c in generated:                          scores.append(-0.5)
                elif c[:4] in generated:                      scores.append(-1.2)
                else:                                         scores.append(-3.5)
            if len(set(scores)) == 1:
                scores[0] += 1.0
            lp_t      = torch.tensor(scores, dtype=torch.float32)
            conf_dist = F.softmax(lp_t, dim=0).tolist()
            pred_idx  = int(torch.argmax(lp_t).item())
            return choices[pred_idx], conf_dist[pred_idx], conf_dist, scores
        except Exception as e:
            print(f"  HF scoring failed: {e}")
            return choices[0], 1/len(choices), [1/len(choices)]*len(choices), [-1.0]*len(choices)

    def score_choice(self, prompt: str, choice: str) -> float:
        choices = self._extract_choices(prompt)
        if not choices: return -2.0
        _, _, dist, _ = self.score_choices(prompt, choices)
        for i, c in enumerate(choices):
            if c.lower() == choice.lower():
                return math.log(max(dist[i], 1e-10))
        return -2.0

    def _extract_choices(self, prompt: str) -> List[str]:
        for line in prompt.split('\n'):
            if 'options:' in line.lower():
                part = line.split(':', 1)[-1]
                return [c.strip() for c in part.split('/') if c.strip()]
        return []

    def generate_reasoning(
        self, question: str, choices: List[str], system_prompt: str = ""
    ) -> str:
        opts   = " / ".join(choices)
        prompt = (
            f"{system_prompt}\n\nQuestion: {question}\n"
            f"Think step by step then answer ({opts}).\n\nThought:"
        )
        try:
            return f"Thought:{self._generate(prompt, 120)}"
        except Exception as e:
            return f"Thought: [unavailable: {e}]"

    @staticmethod
    def build_scoring_prompt(question: str, choices: List[str], few_shot: str = "") -> str:
        return OllamaClient.build_scoring_prompt(question, choices, few_shot)


# ── Mock client ───────────────────────────────

class MockHFApiClient:
    """Mock with keyword heuristics. Always works, no API needed."""

    def __init__(self, model_name: str = "mock"):
        self.model_name = model_name
        print(f"Using MOCK client → {model_name}")

    def score_choices(self, prompt: str, choices: List[str]) -> Tuple[str, float, List[float], List[float]]:
        import hashlib
        p = prompt.lower()
        scores = []

        for choice in choices:
            c  = choice.lower().strip()
            lp = -2.5

            if c == "high":
                hits = sum(1 for w in ["old","accident","smoker: yes","bmi 3","bmi 4","claim"] if w in p)
                lp = -0.2 if hits >= 3 else -1.2 if hits >= 1 else -3.0
            elif c == "low":
                hits = sum(1 for w in ["young","0 accident","smoker: no","bmi 2","no claim"] if w in p)
                lp = -0.2 if hits >= 2 else -1.2 if hits >= 1 else -3.0
            elif c == "medium":
                lp = -1.5
            elif c == "suspicious":
                hits = sum(1 for w in ["days after","inconsisten","not filed","prior claim"] if w in p)
                lp = -0.3 if hits >= 2 else -2.0
            elif c == "not suspicious":
                hits = sum(1 for w in ["420 days","0 claim","police report: filed"] if w in p)
                lp = -0.3 if hits >= 2 else -2.0
            elif c == "compliant":
                hits = sum(1 for w in ["age 35","age 30","disclosed"] if w in p)
                lp = -0.4 if hits >= 1 else -2.0
            elif c == "non-compliant":
                hits = sum(1 for w in ["age 22","not disclosed","15 days","120,000"] if w in p)
                lp = -0.4 if hits >= 1 else -2.0

            seed = int(hashlib.md5(f"{p[:40]}{c}".encode()).hexdigest(), 16) % 1000
            lp += (seed / 1000.0 - 0.5) * 0.25
            scores.append(lp)

        lp_t = torch.tensor(scores, dtype=torch.float32)
        dist = F.softmax(lp_t, dim=0).tolist()
        idx  = int(torch.argmax(lp_t).item())
        return choices[idx], dist[idx], dist, scores

    def score_choice(self, prompt: str, choice: str) -> float:
        choices = self._extract_choices(prompt)
        if not choices: return -2.0
        _, _, dist, _ = self.score_choices(prompt, choices)
        for i, c in enumerate(choices):
            if c.lower() == choice.lower():
                return math.log(max(dist[i], 1e-10))
        return -2.0

    def _extract_choices(self, prompt: str) -> List[str]:
        for line in prompt.split('\n'):
            if 'options:' in line.lower():
                part = line.split(':', 1)[-1]
                return [c.strip() for c in part.split('/') if c.strip()]
        return []

    def generate_reasoning(self, question: str, choices: List[str], system_prompt: str = "") -> str:
        return (
            f"Thought: Analyzing key actuarial risk factors.\n"
            f"Action: Evaluated features against actuarial thresholds.\n"
            f"Answer ({' / '.join(choices)}): [mock prediction]"
        )

    @staticmethod
    def build_scoring_prompt(question: str, choices: List[str], few_shot: str = "") -> str:
        return OllamaClient.build_scoring_prompt(question, choices, few_shot)


# ── Factory ───────────────────────────────────

def get_hf_client(
    api_key:    Optional[str] = None,
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
    use_mock:   bool = False,
    backend:    Optional[str] = None,
):
    """
    Returns best available inference client.
    Priority: ollama > groq > hf > mock

    Override with backend='ollama'|'groq'|'hf'|'mock'
    """
    if use_mock:
        return MockHFApiClient(model_name)

    # Auto-detect or use specified backend
    chosen = backend or detect_backend()
    print(f"  Inference backend: {chosen}")

    if chosen == "ollama":
        try:
            return OllamaClient(model_name)
        except Exception as e:
            print(f"  Ollama failed ({e}), trying next...")
            chosen = "groq" if os.getenv("GROQ_API_KEY") else "mock"

    if chosen == "groq":
        try:
            return GroqClient(model_name)
        except Exception as e:
            print(f"  Groq failed ({e}), falling back to mock")
            return MockHFApiClient(model_name)

    if chosen == "hf":
        key = api_key or os.getenv("HF_API_KEY", "")
        if not key:
            print("  No HF key → using mock")
            return MockHFApiClient(model_name)
        try:
            return HFApiClient(api_key=key, model_name=model_name)
        except Exception as e:
            print(f"  HF failed ({e}) → mock")
            return MockHFApiClient(model_name)

    return MockHFApiClient(model_name)


# ── Quick test ────────────────────────────────

if __name__ == "__main__":
    print(f"\nDetected backend: {detect_backend()}")

    client = get_hf_client()

    FEW_SHOT = (
        "Example: 72yo, 5 accidents, smoker, BMI 41. "
        "Options: high / medium / low\nAnswer: high\n\n"
    )

    prompt = OllamaClient.build_scoring_prompt(
        question = "An insurance applicant is 68 years old, 4 accidents, smoker: yes, BMI 39.2. Classify risk.",
        choices  = ["high", "medium", "low"],
        few_shot = FEW_SHOT,
    )

    pred, conf, dist, scores = client.score_choices(prompt, ["high","medium","low"])
    print(f"\nPredicted:  {pred}")
    print(f"Confidence: {conf:.4f}")
    print(f"Dist:       {[f'{d:.4f}' for d in dist]}")