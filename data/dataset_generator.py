# ─────────────────────────────────────────────
# data/dataset_generator.py
# Generates 500 synthetic actuarial QA pairs
# across 4 task types with ground truth labels
# ─────────────────────────────────────────────

import json
import random
import os
from config import N_QUESTIONS, RANDOM_SEED, DATA_PATH

random.seed(RANDOM_SEED)

# ── Task Type 1: Risk Classification ─────────
# Given applicant features → High / Medium / Low risk

def make_risk_question():
    age        = random.randint(22, 75)
    accidents  = random.randint(0, 5)
    claims     = random.randint(0, 4)
    smoker     = random.choice(["yes", "no"])
    bmi        = round(random.uniform(17.0, 42.0), 1)

    # Deterministic rule-based ground truth
    score = 0
    if age > 60:          score += 2
    elif age > 45:        score += 1
    if accidents >= 3:    score += 3
    elif accidents >= 1:  score += 1
    if claims >= 2:       score += 2
    elif claims == 1:     score += 1
    if smoker == "yes":   score += 2
    if bmi > 35:          score += 2
    elif bmi > 28:        score += 1

    if score >= 6:    label, label_idx = "high",   0
    elif score >= 3:  label, label_idx = "medium", 1
    else:             label, label_idx = "low",    2

    question = (
        f"An insurance applicant is {age} years old, has had {accidents} accidents "
        f"in the past 5 years, filed {claims} claims, is a smoker: {smoker}, "
        f"and has a BMI of {bmi}. "
        f"Classify the risk level as high, medium, or low."
    )
    choices = ["high", "medium", "low"]
    return {
        "task":      "risk_classification",
        "question":  question,
        "choices":   choices,
        "answer":    label,
        "answer_idx": label_idx,
    }

# ── Task Type 2: Fraud Detection ─────────────
# Given claim features → suspicious / not suspicious

def make_fraud_question():
    days_after_policy = random.randint(1, 730)
    claim_amount      = random.randint(500, 50000)
    prior_claims      = random.randint(0, 5)
    inconsistencies   = random.randint(0, 3)
    police_report     = random.choice(["filed", "not filed"])

    score = 0
    if days_after_policy < 30:   score += 3
    elif days_after_policy < 90: score += 1
    if claim_amount > 30000:     score += 2
    elif claim_amount > 15000:   score += 1
    if prior_claims >= 3:        score += 2
    elif prior_claims >= 1:      score += 1
    if inconsistencies >= 2:     score += 3
    elif inconsistencies == 1:   score += 1
    if police_report == "not filed" and claim_amount > 5000:
        score += 2

    label     = "suspicious" if score >= 5 else "not suspicious"
    label_idx = 0 if label == "suspicious" else 1

    question = (
        f"A claim was filed {days_after_policy} days after policy activation. "
        f"Claimed amount: ${claim_amount:,}. Prior claims: {prior_claims}. "
        f"Document inconsistencies found: {inconsistencies}. "
        f"Police report: {police_report}. "
        f"Is this claim suspicious or not suspicious?"
    )
    choices = ["suspicious", "not suspicious"]
    return {
        "task":       "fraud_detection",
        "question":   question,
        "choices":    choices,
        "answer":     label,
        "answer_idx": label_idx,
    }

# ── Task Type 3: Premium Estimation ──────────
# Given features → low / medium / high premium bracket

def make_premium_question():
    age          = random.randint(18, 70)
    coverage     = random.choice(["basic", "standard", "comprehensive"])
    health_score = random.randint(1, 10)
    region       = random.choice(["urban", "suburban", "rural"])
    vehicle_age  = random.randint(0, 20)

    score = 0
    if age > 55:           score += 3
    elif age > 40:         score += 1
    elif age < 25:         score += 2
    if coverage == "comprehensive": score += 3
    elif coverage == "standard":    score += 1
    if health_score < 4:   score += 2
    elif health_score < 7: score += 1
    if region == "urban":  score += 2
    elif region == "suburban": score += 1
    if vehicle_age < 2:    score += 1
    elif vehicle_age > 10: score += 1

    if score >= 7:   label, label_idx = "high",   0
    elif score >= 4: label, label_idx = "medium", 1
    else:            label, label_idx = "low",    2

    question = (
        f"An applicant aged {age} is requesting {coverage} coverage. "
        f"Health score: {health_score}/10. Region: {region}. "
        f"Vehicle age: {vehicle_age} years. "
        f"What premium bracket applies — high, medium, or low?"
    )
    choices = ["high", "medium", "low"]
    return {
        "task":       "premium_estimation",
        "question":   question,
        "choices":    choices,
        "answer":     label,
        "answer_idx": label_idx,
    }

# ── Task Type 4: Policy Compliance ───────────
# Given scenario → compliant / non-compliant

def make_compliance_question():
    policy_age_min  = random.choice([18, 21, 25])
    applicant_age   = random.randint(16, 30)
    coverage_amount = random.randint(10000, 200000)
    max_coverage    = random.choice([50000, 100000, 150000])
    waiting_period  = random.randint(0, 180)    # days
    min_waiting     = random.choice([30, 60, 90])
    pre_existing    = random.choice(["disclosed", "not disclosed"])
    policy_requires = "disclosure"

    violations = 0
    if applicant_age < policy_age_min:   violations += 1
    if coverage_amount > max_coverage:   violations += 1
    if waiting_period < min_waiting:     violations += 1
    if pre_existing == "not disclosed":  violations += 1

    label     = "non-compliant" if violations > 0 else "compliant"
    label_idx = 0 if label == "non-compliant" else 1

    question = (
        f"Policy requires minimum age {policy_age_min}, maximum coverage "
        f"${max_coverage:,}, waiting period of {min_waiting} days, and "
        f"pre-existing condition {policy_requires}. "
        f"Applicant: age {applicant_age}, requested coverage ${coverage_amount:,}, "
        f"waiting period {waiting_period} days, pre-existing condition {pre_existing}. "
        f"Is this application compliant or non-compliant?"
    )
    choices = ["compliant", "non-compliant"]
    return {
        "task":       "policy_compliance",
        "question":   question,
        "choices":    choices,
        "answer":     label,
        "answer_idx": label_idx,
    }

# ── Generator mapping ─────────────────────────
GENERATORS = [
    make_risk_question,
    make_fraud_question,
    make_premium_question,
    make_compliance_question,
]

def generate_dataset(n=N_QUESTIONS):
    """
    Generate n actuarial QA pairs, balanced across 4 task types.
    Returns list of dicts with keys:
        id, task, question, choices, answer, answer_idx
    """
    dataset = []
    per_task = n // len(GENERATORS)

    for gen in GENERATORS:
        for _ in range(per_task):
            item = gen()
            item["id"] = len(dataset)
            dataset.append(item)

    # Shuffle so task types are interleaved
    random.shuffle(dataset)

    # Re-assign sequential ids after shuffle
    for i, item in enumerate(dataset):
        item["id"] = i

    return dataset

def save_dataset(dataset, path=DATA_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"Saved {len(dataset)} questions to {path}")

def load_dataset(path=DATA_PATH):
    with open(path, "r") as f:
        return json.load(f)

def split_dataset(dataset, n_test=100, n_calib=100, seed=RANDOM_SEED):
    """
    Split into train / calibration / test sets.
    train  : used for VIB layer training
    calib  : used for temperature scaling calibration
    test   : held-out evaluation for all UQ methods
    """
    random.seed(seed)
    shuffled = dataset.copy()
    random.shuffle(shuffled)

    test  = shuffled[:n_test]
    calib = shuffled[n_test : n_test + n_calib]
    train = shuffled[n_test + n_calib:]

    print(f"Split → train:{len(train)}  calib:{len(calib)}  test:{len(test)}")
    return train, calib, test

# ── Quick test ────────────────────────────────
if __name__ == "__main__":
    dataset = generate_dataset(N_QUESTIONS)
    save_dataset(dataset)

    # Show one example per task type
    seen = set()
    for item in dataset:
        if item["task"] not in seen:
            seen.add(item["task"])
            print(f"\n{'─'*60}")
            print(f"Task:     {item['task']}")
            print(f"Question: {item['question']}")
            print(f"Choices:  {item['choices']}")
            print(f"Answer:   {item['answer']}")
        if len(seen) == 4:
            break

    train, calib, test = split_dataset(dataset)
    print(f"\nTask distribution in test set:")
    from collections import Counter
    print(Counter(x["task"] for x in test))
