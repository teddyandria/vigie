# Entraîne le modèle de crédit sur la référence, puis mesure l'importance
# de chaque variable. L'outil "importance" de l'agent lira ce résultat.
#
# Sorties (dans data/, régénérables) :
#   - data/model.joblib      : le modèle entraîné
#   - data/importances.json  : l'importance de chaque variable

import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

CIBLE = "class"  # ce qu'on veut prédire : "good" ou "bad" payeur


def construire_modele(colonnes_texte: list) -> Pipeline:
    """Assemble la préparation des données et le modèle en un seul objet."""
    # Le modèle ne comprend que des nombres. Le OneHotEncoder transforme
    # chaque colonne texte en plusieurs colonnes 0/1, une par modalité :
    # purpose="radio/tv" devient purpose_radio/tv=1, purpose_new car=0, etc.
    # handle_unknown="ignore" : une modalité jamais vue (comme dans le lot 05)
    # ne fait pas planter le modèle, elle est juste mise à 0 partout.
    preparation = ColumnTransformer(
        [("texte", OneHotEncoder(handle_unknown="ignore"), colonnes_texte)],
        remainder="passthrough",  # les colonnes numériques passent telles quelles
    )

    # Une forêt aléatoire : plein de petits arbres de décision qui votent.
    # Solide, et elle ne demande presque aucun réglage.
    foret = RandomForestClassifier(n_estimators=300, random_state=42)

    # Le Pipeline enchaîne les deux étapes : on lui donne les données brutes,
    # il les prépare puis les passe au modèle.
    return Pipeline([("preparation", preparation), ("modele", foret)])


def main():
    reference = pd.read_csv("data/reference.csv")

    X = reference.drop(columns=CIBLE)  # les informations sur le client
    y = reference[CIBLE]               # la réponse à prédire

    colonnes_texte = X.select_dtypes(exclude="number").columns.tolist()

    # On garde 25 % de côté pour tester le modèle sur des clients
    # qu'il n'a pas vus pendant l'entraînement. stratify=y garde la même
    # proportion de bons et mauvais payeurs dans les deux parties.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    modele = construire_modele(colonnes_texte)
    modele.fit(X_train, y_train)

    # L'AUC mesure si le modèle classe bien les clients :
    # 0.5 = pile ou face, 1.0 = parfait. Sur ce jeu, 0.75-0.80 est correct.
    proba = modele.predict_proba(X_test)[:, list(modele.classes_).index("bad")]
    auc = roc_auc_score(y_test == "bad", proba)
    print(f"AUC sur les données de test : {auc:.3f}")

    # Importance par permutation : pour chaque variable, on mélange
    # ses valeurs au hasard et on regarde de combien le modèle se dégrade.
    # Si le modèle devient bien pire, la variable comptait beaucoup.
    # Si rien ne change, elle ne comptait pas.
    # On le fait sur les variables d'origine (pas les colonnes 0/1),
    # donc on obtient UN chiffre par variable, ce que veut l'agent.
    resultat = permutation_importance(
        modele, X_test, y_test, scoring="roc_auc", n_repeats=10, random_state=42
    )

    importances = {
        nom: round(float(valeur), 4)
        for nom, valeur in zip(X.columns, resultat.importances_mean)
    }
    # De la plus importante à la moins importante.
    importances = dict(sorted(importances.items(), key=lambda kv: kv[1], reverse=True))

    # On sauvegarde le modèle entraîné sur toute la référence pour la suite
    # du projet, et les importances pour l'outil de l'agent.
    modele.fit(X, y)
    joblib.dump(modele, "data/model.joblib")

    with open("data/importances.json", "w", encoding="utf-8") as f:
        json.dump(importances, f, indent=4)

    print("Importances (perte d'AUC quand on mélange la variable) :")
    for nom, valeur in importances.items():
        print(f"  {nom:<25} {valeur:+.4f}")


if __name__ == "__main__":
    main()
