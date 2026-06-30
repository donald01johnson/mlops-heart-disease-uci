import pandas as pd
from ucimlrepo import fetch_ucirepo

def main():
    # Fetch Heart Disease UCI dataset (id=45)
    heart_disease = fetch_ucirepo(id=45)

    X = heart_disease.data.features
    y = heart_disease.data.targets

    # Combine features and target into a single DataFrame
    df = pd.concat([X, y], axis=1)

    # The target column is typically named 'num' (0 = no disease, 1-4 = disease)
    # Create a binary target: 0 -> 0, 1-4 -> 1
    if "num" in df.columns:
        df["target"] = (df["num"] > 0).astype(int)

    # Save to data/cleveland_heart.csv
    df.to_csv("data/cleveland_heart.csv", index=False)

if __name__ == "__main__":
    main()
