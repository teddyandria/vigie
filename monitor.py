from psi import psi_lot, construire_bacs
import pandas as pd
import json


SEUIL_PSI = 0.25

def main():

    #lire reference.csv et construire les bacs
    reference = pd.read_csv("data/reference.csv")

    bacs = construire_bacs(reference)
    alertes = {}

    for i in range(1, 9):
        lot = pd.read_csv(f"data/batches/day_{i:02d}.csv")

        resultats = psi_lot(bacs, lot)
        #s'il y a des colonnes avec un PSI supérieur au seuil, on les garde dans col_to_keep

        derives = {k: round(v, 3) for k, v in resultats.items() if v > SEUIL_PSI}

        if derives:
            alertes[f"day_{i:02d}"] = derives

    with open("data/alerts.json", "w", encoding="utf-8") as f:
        json.dump(alertes, f, indent=4)

#dunder permet d'importer le module monitor.py sans exécuter le code entier
if __name__ == "__main__":
    main()
