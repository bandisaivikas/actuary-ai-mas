import os
import pandas as pd
import random

def get_real_test_set(csv_path="data/insurance_claims.csv", n_samples=100, seed=42):
    """
    Loads the real Auto Insurance Fraud Dataset from Kaggle,
    converts it into the format expected by the Orchestrator,
    and returns a list of dictionaries.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}. Please download it first.")

    df = pd.read_csv(csv_path)
    
    # Shuffle the dataframe
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Take the first n_samples (or all if n_samples is larger than the dataset)
    if n_samples is not None:
        df = df.head(n_samples)
        
    dataset = []
    
    for i, row in df.iterrows():
        # Map target variable
        # 'Y' means fraud reported -> suspicious
        # 'N' means no fraud reported -> not suspicious
        ground_truth = "suspicious" if row['fraud_reported'] == 'Y' else "not suspicious"
        
        # Build a natural language scenario
        # We try to use the most relevant features for an investigator
        gender = "male" if row['insured_sex'] == 'MALE' else "female"
        age = row['age']
        incident = row['incident_type'].lower()
        collision = str(row['collision_type']).replace('?', 'Unknown')
        severity = row['incident_severity'].lower()
        claim_amount = row['total_claim_amount']
        premium = row['policy_annual_premium']
        months_customer = row['months_as_customer']
        authorities = row['authorities_contacted']
        witnesses = row['witnesses']
        
        question = (
            f"An auto insurance claim has been filed by a {age}-year-old {gender} "
            f"who has been a customer for {months_customer} months, paying an annual premium of ${premium:.2f}. "
            f"The reported incident was a {incident} ({collision}) resulting in {severity}. "
            f"Authorities contacted: {authorities}. Number of witnesses: {witnesses}. "
            f"The total claimed amount is ${claim_amount}. "
            "Based on these details, classify this claim for potential fraud."
        )
        
        record = {
            "id": i,
            "task": "fraud_detection",
            "question": question,
            "choices": ["suspicious", "not suspicious"],
            "ground_truth": ground_truth
        }
        
        dataset.append(record)
        
    return dataset

if __name__ == "__main__":
    # Test the loader
    data = get_real_test_set(n_samples=2)
    for d in data:
        print(d["question"])
        print(f"Ground Truth: {d['ground_truth']}\n")
