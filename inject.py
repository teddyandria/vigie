import pandas as pd
from pandas import DataFrame
import numpy as np

def day_03(df) -> DataFrame:
    #multiplier credit_amount par 100
    df["credit_amount"] = df["credit_amount"] * 100

    return df

def day_04(df, rng) -> DataFrame:
    #données manquantes.
    frac = df.sample(frac=0.4, random_state=rng)
    df.loc[frac.index, "duration"] = np.nan

    print(df["duration"].isna().mean())
    return df

def day_05(df, rng) -> DataFrame:
    #day_05, modalité inconnue : une partie de "purpose" remplacée par une valeur jamais vue ;
    frac = df.sample(frac=0.3, random_state=rng)
    df.loc[frac.index, "purpose"] = "new_value_panne"

    return df

def day_06(df) -> DataFrame:
    #day_06, vraie dérive : les clients ont huit ans de plus, et leur ancienneté d'emploi augmente avec ;
    employment_mapping = {
        "unemployed": "unemployed",
        "<1": "8<=X<9",
        "1<=X<4": "9<=X<12",
        "4<=X<7": "12<=X<15",
        ">=7": ">=15"}

    df["employment"] = df["employment"].map(employment_mapping)
    df["age"] = df["age"] + 8

    print(df["employment"].value_counts())
    print(df["employment"].isna().sum())
    return df

def day_07(df, rng_panne) -> DataFrame:
    #day_07, fausse alerte : 80 % de num_dependents passés à 2, une variable qui compte très peu pour le modèle.
    frac = df.sample(frac=0.8, random_state=rng_panne)
    df.loc[frac.index, "num_dependents"] = 2
    return df

def main():

    production = pd.read_csv("data/production.csv")
    rng = np.random.default_rng(seed=42)
    rng_panne = np.random.default_rng(seed=43)
    for i in range(1, 9):
        parcelle = production.sample(n=250, random_state=rng, replace=True)
        day = f"day_{i:02d}"
        parcelle = parcelle.reset_index(drop=True)

        #print(parcelle.index.duplicated().sum())
        if day == "day_03":
            parcelle = day_03(parcelle)
            parcelle.to_csv("data/batches/day_03.csv", index= False)

        elif day == "day_04":
            parcelle = day_04(parcelle, rng_panne)
            parcelle.to_csv("data/batches/day_04.csv", index= False)

        elif day == "day_05":
            parcelle = day_05(parcelle, rng_panne)
            parcelle.to_csv("data/batches/day_05.csv", index= False)

        elif day == "day_06":
            parcelle = day_06(parcelle)
            parcelle.to_csv("data/batches/day_06.csv", index= False)

        elif day == "day_07":
            parcelle = day_07(parcelle, rng_panne)
            parcelle.to_csv("data/batches/day_07.csv", index= False)

main()