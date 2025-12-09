import pandas as pd
import numpy as np
import os


def load_raw(path):
    print(f"Loading raw data from {path} ...")
    df = pd.read_csv(path, sep="|", low_memory=False)
    print(f"Loaded {df.shape[0]:,} rows and {df.shape[1]} columns.")
    return df


def clean_numeric(df):
    print("Cleaning numeric columns...")

    numeric_cols = [
        "mmcode", "RegistrationYear", "Cylinders", "cubiccapacity",
        "kilowatts", "NumberOfDoors", "CapitalOutstanding",
        "SumInsured", "CalculatedPremiumPerTerm",
        "TotalPremium", "TotalClaims"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fix negatives in CapitalOutstanding
    if "CapitalOutstanding" in df.columns:
        df.loc[df["CapitalOutstanding"] < 0, "CapitalOutstanding"] = 0

    # Remove unrealistic mmcode
    if "mmcode" in df.columns:
        df.loc[df["mmcode"] > 200000, "mmcode"] = np.nan

    return df


def clean_dates(df):
    print("Cleaning date columns...")

    if "TransactionMonth" in df.columns:
        df["TransactionMonth"] = pd.to_datetime(
            df["TransactionMonth"], errors="coerce"
        )

    return df


def clean_categoricals(df):
    print("Cleaning categorical columns...")

    # Strip whitespace
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip().replace({"": np.nan})

    # Normalize Gender
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].str.upper()
        df["Gender"] = df["Gender"].replace({
            "MALE": "M", "FEMALE": "F",
            "M": "M", "F": "F"
        })

    return df


def handle_missing(df):
    print("Handling missing values...")

    # Drop rows missing essential values
    df = df.dropna(subset=["TransactionMonth", "TotalPremium"])

    # Fill categorical NA with "Unknown"
    cat_cols = df.select_dtypes(include="object").columns
    df[cat_cols] = df[cat_cols].fillna("Unknown")

    # Fill numeric with median
    num_cols = df.select_dtypes(include=["float64", "int64"]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    return df


def create_features(df):
    print("Creating derived features...")

    # Remove rows with zero premium
    df = df[df["TotalPremium"] > 0]

    # Create loss ratio
    df["lossratio"] = df["TotalClaims"] / df["TotalPremium"]

    # Binary has_claim
    df["has_claim"] = (df["TotalClaims"] > 0).astype(int)

    # Cap extreme loss ratios
    cap_value = df["lossratio"].quantile(0.99)
    df.loc[df["lossratio"] > cap_value, "lossratio"] = cap_value

    return df


def clean_outliers(df):
    print("Capping outliers...")

    num_cols = [
        "SumInsured", "TotalPremium", "TotalClaims", "CalculatedPremiumPerTerm"
    ]

    for col in num_cols:
        if col in df.columns:
            upper = df[col].quantile(0.99)
            df.loc[df[col] > upper, col] = upper

    return df


def save_clean(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Cleaned data saved to {path}.")


def main():
    raw_path = "data/raw/MachineLearningRating_v3.txt"
    clean_path = "data/interim/cleaned.csv"

    df = load_raw(raw_path)
    df = clean_numeric(df)
    df = clean_dates(df)
    df = clean_categoricals(df)
    df = handle_missing(df)
    df = create_features(df)
    df = clean_outliers(df)
    save_clean(df, clean_path)


if __name__ == "__main__":
    main()
