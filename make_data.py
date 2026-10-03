from pathlib import Path
from sklearn.datasets import fetch_openml
import numpy as np


def save_data(reference, production):

    reference.to_csv("data/reference.csv", index=False)
    production.to_csv("data/production.csv", index=False)

def batch(production):
    Path("data/batches/").mkdir(parents=True, exist_ok=True)
    
    rng = np.random.default_rng(seed=42)
    for i in range(1, 9):
        parcelle = production.sample(n=250, random_state=rng, replace=True)
        day = f"day_{i:02d}"
        parcelle.to_csv("data/batches/" + day + ".csv", index= False)


def main():
    credit = fetch_openml(name="credit-g", version=1, as_frame=True, parser="pandas")
    df = credit.frame

    shuffled = df.sample(n=1000, random_state=42)

    reference = shuffled.iloc[:600]
    production = shuffled.iloc[600:]

    reference = reference.reset_index(drop=True)

    save_data(reference, production)
    batch(production)

main()




