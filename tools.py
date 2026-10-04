#ecrire cinq outils qui prennent un jour, et parfois une variable et renvoie un dictionnaire.
#écris cinq outils qui prennent un jour, et parfois une variable, et renvoient un petit dictionnaire :
# • le PSI de chaque variable ;
# • la moyenne et la médiane d'une variable, référence contre lot ;
# • le taux de valeurs manquantes par variable ;
# • les modalités jamais vues dans la référence ;
# • l'importance d'une variable pour le modèle.

#langchain permet de transformer une fonction Python en outil qu'un LLM peut appeler. langchain lit 3 choses : son nom, ses paramètres et leurs types, docstring.

from langchain_core.tools import tool
from numpy import number
from psi import psi_lot, construire_bacs
import pandas as pd
from functools import lru_cache
from pathlib import Path

@lru_cache
def _load_data():
    reference = pd.read_csv("data/reference.csv")
    bacs = construire_bacs(reference)

    return reference, bacs

@lru_cache
def _load_lot(day):

    path_location = Path(f"data/batches/{day}.csv")
    lot_exist = path_location.exists()

    #évite les bugs si une le jour est endehors de la plage (01 à 08)
    if lot_exist:   
        lot = pd.read_csv(path_location)
        return lot
    return None


@tool
def compute_psi(day: str) -> dict:
    """Calcule le PSI de chaque variable, référence contre lot.
    Si le psi vaut moins de 0,1, c'est considéré comme normal. De 0,1 à 0,25, il faut surveiller la variable en question, supérieur à 0,25, il y a un changement fort.
    Un PSI élevé indique qu'une variable a changé, pas pourquoi ni si elle compte pour le modèle.

    À utiliser en premier lieu, après une alerte, pour savoir quelles variables examiner avec les autres outils.

    Args:
        day: nom du lot, au format "day_03".
    """
    _, bacs = _load_data()
    lot = _load_lot(day)

    if lot is None:
        return {"erreur": f"jour inconnu : {day}, format attendu : day_01 à day_08"}
    
    psi = psi_lot(bacs, lot)

    psi_per_variable = {k: round(v, 3) for k, v in psi.items()}

    result = {
        "jour" : day,
        "psi" : psi_per_variable,
    }

    return result


#moyenne et médiane, reference contre lot
#permet de savoir de combien la variable à bougé.
@tool
def compare_mean_median(day: str, variable: str) -> dict:
    """Compare la moyenne et la médiane d'une variable entre la référence et le lot du jour. 
    Si seule la moyenne bouge, quelques valeurs extrêmes en sont la cause. Si la moyenne et la médiane bougent ensemble, c'est toute la distribution qui est décalée.
    À utiliser après le PSI, pour mesurer l'ampleur d'un écart sur une variable signalée.
    Un rapport proche de 100, 1000 ou 0,01 suggère un problème d'unité (centimes au lieu d'euros, par exemple)
    
    Args:
        day: nom du lot, au format "day_03".
        variable: nom d'une variable numérique, par exemple "credit_amount" ou "age". Pour une variable texte, utiliser l'outil des modalités inconnues.
    """

    #charger
    reference, _ = _load_data()
    lot = _load_lot(day)

    #vérification
    if lot is None:
        return {"erreur": f"jour inconnu : {day}, format attendu : day_01 à day_08"}

    if variable not in reference.columns:
        return {"erreur": f"variable inconnue : {variable}."}

    if not pd.api.types.is_numeric_dtype(reference[variable]):
        return {"erreur": f"le format de variable n'est pas accepté.Format attendu : numerique."}
    
    #calcule
    var_from_ref = reference[variable]
    var_from_lot = lot[variable]
    #renvoyer

    median_ref = var_from_ref.median(numeric_only=True)
    median_lot = var_from_lot.median(numeric_only=True)

    mean_ref = var_from_ref.mean()
    mean_lot = var_from_lot.mean()

    rapport_median = median_lot / median_ref
    rapport_mean = mean_lot / mean_ref

    return {
        "jour": day,
        "variable" : variable,
        "reference" : {
            "moyenne": mean_ref,
            "mediane" : median_ref,
        },
        "lot": {
            "moyenne": mean_lot,
            "mediane" : median_lot,
        },
        "rapport": {
            "moyenne": rapport_mean,
            "mediane" : rapport_median,
        }
    }