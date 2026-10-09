#langchain est un framework qui fournit des outils pour connecter des modèles de langage à des sources de données et des outils externes. Il permet de créer des agents capables d'effectuer des tâches complexes en combinant plusieurs modèles et outils.

#pydantic est une bibliothèque qui permet de définir des modèles de données avec validation et sérialisation. Elle est utilisée ici pour définir le format de la réponse finale de l'agent.

from dotenv import load_dotenv
from langchain.agents import create_agent
from tools import TOOLS
from pydantic import BaseModel, Field
from typing import Literal



load_dotenv()  # charge les variables d'environnement depuis le fichier .env


class Diagnostic(BaseModel):
    cause: Literal["bug_unite", "donnees_manquantes", "modalite_inconnue", "derive_population", "fausse_alerte"] = Field(..., description="bug_unite : un bug dans l'unité de mesure a provoqué la dérive, donnees_manquantes : des données sont manquantes dans le lot, modalite_inconnue : une modalité inconnue est apparue dans le lot, derive_population : la population a changé et le modèle n'est plus adapté, fausse_alerte : le PSI est supérieur à 0,25, la derive est réelle mais n'a pas d'impact sur le modèle")
    variables: list[str] = Field(..., description="la liste des variables en cause")
    gravite : Literal["mineure", "majeure", "critique"] = Field(..., description="mineure : impact faible sur le modèle, majeure : impact significatif sur le modèle, critique : impact majeur sur le modèle")
    action: Literal["corriger_pipeline", "reentrainer_model", "ignorer"] = Field(..., description="corriger_pipeline : corriger le pipeline de scoring, reentrainer_model : réentraîner le modèle, ignorer : ne rien faire")
    preuves: list[str] = Field(..., description="la liste des preuves qui permettent de trancher, avec les chiffres des outils qui ont été utilisés. Une observation par élément, avec l'outil utilisé et le chiffre obtenu.")

#le system prompt est un texte qui définit le rôle et les instructions de l'agent. Utilisé pour guider le comportement de l'agent et lui fournir un contexte sur la tâche à accomplir.

SYSTEM_PROMPT = """Tu es l'ingénieur d'astreinte d'un modèle de scoring de crédit en production.
Le moniteur vient de lever une alerte de dérive sur un lot quotidien : au moins une variable a un PSI supérieur à 0,25.

Ta mission : trouver la cause de l'alerte et recommander une action. Tu ne corriges rien toi-même.

Méthode :
1. Appelle d'abord get_runbook et suis sa procédure, étape par étape, dans l'ordre.
2. Appelle les outils dont la procédure a besoin pour le lot concerné.
3. Conclus quand la procédure te permet de trancher.

Règles :
- Ne conclus qu'à partir des résultats des outils. N'invente aucun chiffre.
- Ne conclus pas sur le seul PSI : il dit qu'une variable a changé, pas pourquoi.
- Vérifie d'abord les causes techniques, même si la variable compte peu pour le modèle.
- Utilise exactement les mots-clés du runbook pour la cause et l'action.
- Si un outil renvoie une erreur, corrige ton appel et réessaie.

Termine par ton diagnostic : choisis la gravité selon le tableau du runbook, et cite dans les preuves chaque observation décisive avec l'outil utilisé et son chiffre.
"""

graph = create_agent(
    model="openai:gpt-5.5",
    tools=TOOLS,
    response_format=Diagnostic,
    system_prompt=SYSTEM_PROMPT
)

if __name__ == "__main__":
    resultat = graph.invoke({"messages": [{"role": "user", "content": "Alerte sur day_03 : ..."}]})
    print(resultat["structured_response"])