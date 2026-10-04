#psi = Polulation Stability Index, une mesure de dérive de population. Plus le psi est élevé, plus la population a changé par rapport à la population de référence.
# on le calcule variable par variable.
# il répond à la question : la repartition de cette colonne a-t-elle changé entre la référence et le lot du jour" ?
# < 0,1 : pas de dérive, 
# 0,1 à 0,25 : dérive modérée donc à surveiller, 
# > 0,25 : dérive importante
# un bac = plage de valeurs, exemple sur age : 18-25, 26-35 etc...

# formule : (cur − ref) × ln(cur / ref).  -----> ln = np.log

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

# Nombre de bacs pour les colonnes numériques.
# 10 bacs = chaque bac contient environ 10 % des clients de la référence.
N_BACS = 10

# Valeur qui remplace une proportion nulle.
# Sans ça, ln(0) et la division par 0 font exploser le calcul.
PLANCHER = 1e-4


# ---------------------------------------------------------------------------
# 1. La formule
# ---------------------------------------------------------------------------

def psi_depuis_proportions(ref_props, cur_props) -> float:
    """Calcule le PSI à partir de deux listes de proportions.

    ref_props[i] et cur_props[i] doivent parler du MÊME bac.
    Exemple : [0.5, 0.3, 0.2] et [0.2, 0.3, 0.5] donnent environ 0.55.
    """
    ref = np.asarray(ref_props, dtype=float)
    cur = np.asarray(cur_props, dtype=float)

    # Un bac vide a une proportion de 0 : on le remonte au plancher.
    ref = np.maximum(ref, PLANCHER)
    cur = np.maximum(cur, PLANCHER)

    # Un terme par bac, puis on additionne tous les termes.
    termes = (cur - ref) * np.log(cur / ref)
    return float(termes.sum())


# ---------------------------------------------------------------------------
# 2. Colonnes numériques : bacs par quantiles
# ---------------------------------------------------------------------------

def _frontieres_numeriques(ref_col: Series) -> np.ndarray:
    """Calcule les frontières des bacs, UNE SEULE FOIS, sur la référence."""
    # Les niveaux 10 %, 20 %, ..., 90 % : ce sont les coupures intérieures.
    niveaux = np.linspace(0, 1, N_BACS + 1)[1:-1]
    coupures = np.quantile(ref_col.dropna(), niveaux)

    # Pour une colonne avec peu de valeurs différentes (ex. num_dependents
    # qui ne vaut que 1 ou 2), plusieurs coupures sont identiques.
    # On ne garde que les coupures uniques, sinon pd.cut plante.
    coupures = np.unique(coupures)

    # On ajoute -infini au début et +infini à la fin : ainsi, une valeur
    # plus petite ou plus grande que tout ce qu'on a vu dans la référence
    # (ex. les montants x100 du lot 03) tombe quand même dans un bac.
    return np.concatenate(([-np.inf], coupures, [np.inf]))


def _proportions_numeriques(col: Series, frontieres: np.ndarray) -> np.ndarray:
    """Range chaque valeur dans son bac et renvoie la part de chaque bac.

    Le dernier élément est la part de valeurs manquantes (NaN).
    """
    n_bacs = len(frontieres) - 1

    # pd.cut donne, pour chaque valeur, le numéro de son bac (0, 1, 2...).
    # Une valeur manquante reçoit NaN.
    numeros = pd.cut(col, bins=frontieres.tolist(), labels=False)

    # On compte combien de valeurs tombent dans chaque bac.
    comptes = np.bincount(numeros.dropna().astype(int), minlength=n_bacs)

    # Bac supplémentaire pour les valeurs manquantes : c'est lui qui
    # rendra visible la panne du lot 04.
    comptes = np.append(comptes, col.isna().sum())

    # Des comptes aux proportions : on divise par le nombre total de lignes.
    return comptes / len(col)


# ---------------------------------------------------------------------------
# 3. Colonnes texte : un bac par modalité, plus "autre"
# ---------------------------------------------------------------------------

def _modalites_texte(ref_col: Series) -> list:
    """Liste des modalités vues dans la référence, triée pour avoir un ordre fixe."""
    return sorted(ref_col.dropna().unique())


def _proportions_texte(col: Series, modalites: list) -> np.ndarray:
    """Renvoie la part de chaque modalité, puis "autre", puis "manquant".

    "autre" = toute valeur jamais vue dans la référence (ex. le lot 05).
    """
    connue = col.isin(modalites)
    manquante = col.isna()
    inconnue = ~connue & ~manquante

    # value_counts compte chaque modalité ; reindex les remet dans l'ordre
    # de la référence et met 0 pour une modalité absente du lot.
    comptes = col[connue].value_counts().reindex(modalites, fill_value=0).to_numpy()

    comptes = np.append(comptes, [inconnue.sum(), manquante.sum()])
    return comptes / len(col)


# ---------------------------------------------------------------------------
# 4. Assembler : construire les bacs une fois, puis calculer pour chaque lot
# ---------------------------------------------------------------------------

def construire_bacs(reference: DataFrame) -> dict:
    """Prépare, pour chaque colonne, ses bacs et ses proportions de référence.

    On l'appelle UNE fois. Le résultat sert ensuite pour tous les lots.
    exemple de retour : {"age": {"type": "numerique", "frontieres": [...], "ref": [...]}, ...}
    """
    bacs = {}
    for nom in reference.columns:
        ref_col = reference[nom]

        if pd.api.types.is_numeric_dtype(ref_col):
            frontieres = _frontieres_numeriques(ref_col)
            bacs[nom] = {
                "type": "numerique",
                "frontieres": frontieres,
                "ref": _proportions_numeriques(ref_col, frontieres),
            }
        else:
            modalites = _modalites_texte(ref_col)
            bacs[nom] = {
                "type": "texte",
                "modalites": modalites,
                "ref": _proportions_texte(ref_col, modalites),
            }
    return bacs


def psi_colonne(bac: dict, cur_col: Series) -> float:
    """PSI d'une colonne du lot, avec les bacs préparés sur la référence."""
    if bac["type"] == "numerique":
        cur = _proportions_numeriques(cur_col, bac["frontieres"])
    else:
        cur = _proportions_texte(cur_col, bac["modalites"])
    return psi_depuis_proportions(bac["ref"], cur)


def psi_lot(bacs: dict, lot: DataFrame) -> dict:
    """PSI de toutes les colonnes d'un lot, de la plus dérivée à la moins dérivée."""
    resultats = {nom: psi_colonne(bac, lot[nom]) for nom, bac in bacs.items()}
    return dict(sorted(resultats.items(), key=lambda kv: kv[1], reverse=True))


# ---------------------------------------------------------------------------
# 5. Démonstration : python psi.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Vérification de la formule sur l'exemple fait à la main : ~0.55.
    print("Exemple à la main :", round(psi_depuis_proportions([0.5, 0.3, 0.2], [0.2, 0.3, 0.5]), 3))

    reference = pd.read_csv("data/reference.csv")
    bacs = construire_bacs(reference)

    # La référence comparée à elle-même doit donner 0 partout.
    print("Référence vs elle-même (max) :", round(max(psi_lot(bacs, reference).values()), 6))

    # Pour chaque lot, on affiche les 3 colonnes qui ont le plus bougé.
    for i in range(1, 9):
        lot = pd.read_csv(f"data/batches/day_{i:02d}.csv")
        top3 = list(psi_lot(bacs, lot).items())[:3]
        print(f"day_{i:02d} :", ", ".join(f"{nom}={valeur:.2f}" for nom, valeur in top3))