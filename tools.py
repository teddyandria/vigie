#écris cinq outils qui prennent un jour, et parfois une variable, et renvoient un petit dictionnaire :
# • le PSI de chaque variable ;
# • la moyenne et la médiane d'une variable, référence contre lot ;
# • le taux de valeurs manquantes par variable ;
# • les modalités jamais vues dans la référence ;
# • l'importance d'une variable pour le modèle.

#langchain permet de transformer une fonction Python en outil qu'un LLM peut appeler. langchain lit 3 choses : son nom, ses paramètres et leurs types, docstring.

import json
from langchain_core.tools import tool
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
        variable: nom d'une variable numérique, par exemple "credit_amount" ou "age". Pour une variable texte, utiliser l'outil : never_seen_modality.
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

@tool
def missing_data_rate(day: str) -> dict:
    """Calcule le taux de valeurs manquantes de chaque variable du lot, comparé à la référence.

    À utiliser quand une variable a un PSI élevé mais que sa moyenne et sa médiane ont peu bougé : le changement vient peut-être de cases vides.

    Ne renvoie que les variables qui ont au moins une valeur manquante dans le lot. Un taux de 0.4 signifie 40 % de cases vides. La référence n'a normalement aucune valeur manquante : un taux élevé et soudain indique en général un problème de collecte ou de pipeline, pas un vrai changement chez les clients. Si "taux" est vide, le lot est complet.

    Args:
        day: nom du lot, au format "day_03".
    """
    reference, _ = _load_data()
    lot = _load_lot(day)

    if lot is None:
        return {"erreur": f"jour inconnu : {day}, format attendu : day_01 à day_08"}

    # isna() marque chaque case vide par True, mean() fait la moyenne par
    # colonne : on obtient la part de cases vides de chaque variable.
    taux_lot = lot.isna().mean()
    taux_ref = reference.isna().mean()

    # On parcourt les variables une par une et on ne garde que celles
    # qui ont au moins une case vide (taux > 0) dans le lot.
    taux = {
        variable: {
            "lot": round(float(t), 3),
            "reference": round(float(taux_ref.get(variable, 0.0)), 3),
        }
        for variable, t in taux_lot.items()
        if t > 0
    }

    return {
        "jour": day,
        "taux": taux,
    }

@tool
def never_seen_modality(day: str) -> dict:
    """Cherche, dans les variables texte du lot, les valeurs qui n'existent pas dans la référence.

    À utiliser quand une variable texte (par exemple "purpose") a un PSI élevé, pour savoir si c'est parce qu'une valeur inconnue est apparue.
    Le modèle n'a jamais appris ces valeurs : il prédit à l'aveugle pour les clients concernés. Une valeur inconnue qui touche beaucoup de lignes d'un coup indique en général un problème technique (nouveau choix dans l'application, faute de frappe, encodage changé), pas un
    vrai changement chez les clients.

    Ne renvoie que les variables qui ont au moins une valeur inconnue, avec le nombre de lignes par valeur et la part du lot touchée (0.3 = 30 %).
    Si "modalites_inconnues" est vide, toutes les valeurs sont connues.
    Les cases vides ne sont pas comptées : voir missing_data_rate.

    Args:
        day: nom du lot, au format "day_03".
    """

    #charger
    reference, _ = _load_data()
    lot = _load_lot(day)

    #verif
    if lot is None:
        return {"erreur": f"jour inconnu : {day}, format attendu : day_01 à day_08"}

    result = {}
    for col in reference.columns:
        # On ne regarde que les colonnes texte : on teste la colonne
        # elle-même (reference[col]), pas son nom.
        if not pd.api.types.is_string_dtype(reference[col]):
            continue

        values_ref = reference[col]
        values_lot = lot[col].dropna()  # une case vide n'est pas une valeur inconnue

        # Le masque : pour chaque ligne du lot, True si la valeur est INCONNUE.
        masque_inconnues = ~values_lot.isin(values_ref)

        # On applique le masque : il ne reste que les lignes inconnues.
        inconnues = values_lot[masque_inconnues]

        # Aucune ligne inconnue dans cette colonne : on passe à la suivante.
        if inconnues.empty:
            continue

        # value_counts compte chaque valeur inconnue ; on le transforme
        # en vrai dictionnaire Python, avec des int au lieu de nombres numpy.
        comptes = {valeur: int(n) for valeur, n in inconnues.value_counts().items()}

        result[col] = {
            "valeurs": comptes,
            "part_du_lot": round(len(inconnues) / len(lot), 3),
        }

    return {
        "jour": day,
        "modalites_inconnues": result,
    }

@lru_cache
def _load_importances():
    #importances calculées par train.py ; None si le fichier n'existe pas encore
    path_location = Path("data/importances.json")
    if not path_location.exists():
        return None
    with open(path_location, encoding="utf-8") as f:
        return json.load(f)

@tool
def feature_importance(variable: str) -> dict:
    """Indique à quel point le modèle s'appuie sur une variable pour prédire.

    À utiliser après avoir trouvé une variable qui a beaucoup changé, pour
    juger si ce changement peut dégrader les prédictions. Une variable qui
    dérive mais compte peu pour le modèle est souvent une fausse alerte.
    Attention : une faible importance ne rend pas acceptable un problème
    technique (unité fausse, valeurs manquantes, modalité inconnue), qu'il
    faut corriger dans tous les cas.

    Renvoie :
    - importance : perte de qualité du modèle (AUC) quand on brouille la
      variable. 0 ou négatif = la variable ne sert pas.
    - rang : 1 = la variable la plus importante du modèle.
    - part : part de l'importance totale du modèle (0.5 = 50 %).
    - niveau : "forte" (part >= 10 %), "moyenne" (>= 3 %) ou "faible".

    Args:
        variable: nom d'une variable du modèle, par exemple "credit_amount".
    """
    importances = _load_importances()

    if importances is None:
        return {"erreur": "importances introuvables : lancer d'abord python train.py"}

    if variable not in importances:
        return {"erreur": f"variable inconnue : {variable}. Variables disponibles : {list(importances)}"}

    # Les importances négatives sont du bruit : on les compte comme 0.
    positives = {nom: max(valeur, 0.0) for nom, valeur in importances.items()}
    total = sum(positives.values())

    # importances.json est déjà trié de la plus importante à la moins importante,
    # donc la position dans la liste donne le rang.
    rang = list(importances).index(variable) + 1
    part = positives[variable] / total if total > 0 else 0.0

    if part >= 0.10:
        niveau = "forte"
    elif part >= 0.03:
        niveau = "moyenne"
    else:
        niveau = "faible"

    return {
        "variable": variable,
        "importance": round(float(importances[variable]), 4),
        "rang": rang,
        "nb_variables": len(importances),
        "part": round(part, 3),
        "niveau": niveau,
    }

@tool
def get_runbook()-> dict:
    """Renvoie la procédure d'astreinte de l'équipe (RUNBOOK.md) pour enquêter sur une alerte de dérive.

    À appeler en premier, au début de chaque enquête, avant tout autre outil.
    Le runbook indique dans quel ordre utiliser les outils, comment
    reconnaître chacune des cinq causes possibles (bug_unite,
    donnees_manquantes, modalite_inconnue, derive_population,
    fausse_alerte) et quelle action recommander (corriger_pipeline,
    reentrainer_model, ignorer). Suivre cette procédure plutôt
    qu'improviser, et reprendre ses mots-clés à l'identique dans la
    conclusion.
    """
    path_location = Path("RUNBOOK.md")
    if not path_location.exists():
        return {"erreur": f"Le fichier : {path_location} n'existe pas."}
    
    texte = path_location.read_text(encoding="utf-8")

    return {"runbook": texte}


def main():
    print(feature_importance.invoke({"variable": "checking_status"}))
    print(feature_importance.invoke({"variable": "num_dependents"}))
    print(feature_importance.invoke({"variable": "montant"}))
    print(missing_data_rate.invoke({"day": "day_04"}))
    print(missing_data_rate.invoke({"day": "day_08"}))
    print(never_seen_modality.invoke({"day": "day_05"}))
    print(never_seen_modality.invoke({"day": "day_01"}))

if __name__ == "__main__":
    main()

TOOLS = [compute_psi, compare_mean_median, missing_data_rate, never_seen_modality, feature_importance, get_runbook]