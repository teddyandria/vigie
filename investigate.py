
from langfuse import get_client
from agent import graph
import argparse
import json
from langfuse.langchain import CallbackHandler

def construire_message(day: str, alerte: dict) -> str:
    """Transforme une alerte de alerts.json en message pour l'agent.

    {"num_dependents": 1.839} devient "num_dependents (PSI 1.839)".
    Partagée avec eval.py, pour que l'évaluation envoie exactement le même message.
    """
    variables = ", ".join(f"{variable} (PSI {psi})" for variable, psi in alerte.items())
    return f"Alerte de dérive sur {day} : {variables}. Enquête et donne ton diagnostic."

def main():
    langfuse_handler = CallbackHandler()  # crée un gestionnaire de callback pour Langfuse, qui permet de suivre les interactions avec l'agent

    # argparse permet de récupérer les arguments passés en ligne de commande. Ici, on attend un argument "day" qui indique le lot à investiguer.
    parser = argparse.ArgumentParser(
        description="Lance l'agent d'astreinte sur un lot et affiche son diagnostic."
    )

    parser.add_argument("day", help="La date du lot à investiguer, au format day_XX, par exemple day_03")

    args = parser.parse_args()


    with open("data/alerts.json", "r", encoding="utf-8") as f:
        alertes = json.load(f)

    # Pas d'alerte : on n'appelle pas l'agent (lot sain ou jour inconnu).
    if args.day not in alertes:
        print(f"Aucune alerte pour : {args.day}")
        return

    msg = construire_message(args.day, alertes[args.day])
    print(msg)

    resultat = graph.invoke(
        {"messages": [{"role": "user", "content": msg}]},
        config={"callbacks":[langfuse_handler]}  # on passe le gestionnaire de callback à l'agent pour suivre les interactions
    )

    # L'enquête : les outils appelés par l'agent, dans l'ordre.
    # Chaque message de l'agent peut contenir des appels d'outils (tool_calls).
    print("\nEnquête :")
    for message in resultat["messages"]:
        for appel in getattr(message, "tool_calls", None) or []:
            print(f"  - {appel['name']}({appel['args']})")

    # Le diagnostic : un objet Diagnostic, affiché champ par champ.
    diagnostic = resultat["structured_response"]
    print("\nDiagnostic :")
    print(f"  cause     : {diagnostic.cause}")
    print(f"  variables : {', '.join(diagnostic.variables)}")
    print(f"  gravité   : {diagnostic.gravite}")
    print(f"  action    : {diagnostic.action}")
    print("  preuves   :")
    for preuve in diagnostic.preuves:
        print(f"    - {preuve}")

    get_client().flush()  # on envoie les événements à Langfuse pour qu'ils soient visibles dans le dashboard
if __name__ == "__main__":
    main()
