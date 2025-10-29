import pandas as pd
import numpy as np
import random

def random_trajectory():
    # Simulate a learning trajectory (e.g., years to acquire new skills)
    return np.round(np.random.uniform(0.5, 3.0), 2)

def random_complexity():
    # Simulate problem-solving complexity progression (0-1 scale)
    return np.round(np.random.beta(2, 2), 2)

def random_adaptability():
    # Simulate adaptability index (number of domains/skills)
    return random.randint(1, 10)

def random_innovation_density():
    # Simulate innovation density (novel projects per year)
    return np.round(np.random.uniform(0, 2), 2)

def random_collaboration():
    # Simulate collaboration patterns (0=solo, 1=team, 0.5=mixed)
    return np.round(np.random.choice([0, 0.5, 1]), 2)

def random_depth_breadth():
    # Simulate knowledge depth vs breadth (0=generalist, 1=specialist)
    return np.round(np.random.beta(2, 2), 2)

def random_success_label():
    # Simulate a success label (1=high performer, 0=average)
    return np.random.choice([0, 1], p=[0.7, 0.3])

def generate_synthetic_cognitive_resumes(n=200):
    data = []
    for _ in range(n):
        row = {
            "learning_velocity": random_trajectory(),
            "problem_solving_complexity": random_complexity(),
            "adaptability_index": random_adaptability(),
            "innovation_density": random_innovation_density(),
            "collaboration_patterns": random_collaboration(),
            "knowledge_depth_vs_breadth": random_depth_breadth(),
            "success_label": random_success_label()
        }
        data.append(row)
    return pd.DataFrame(data)

if __name__ == "__main__":
    df = generate_synthetic_cognitive_resumes(500)
    df.to_csv("synthetic_cognitive_resumes.csv", index=False)
    print("Synthetic dataset saved as synthetic_cognitive_resumes.csv")