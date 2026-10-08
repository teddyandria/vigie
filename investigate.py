
from agent import graph
import argparse
import json

def main():
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

    # On transforme {"num_dependents": 1.839} en "num_dependents (PSI 1.839)".
    variables = ", ".join(f"{variable} (PSI {psi})" for variable, psi in alertes[args.day].items())
    msg = f"Alerte de dérive sur {args.day} : {variables}. Enquête et donne ton diagnostic."
    print(msg)

    resultat = graph.invoke({"messages": [{"role": "user", "content": msg}]})

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


if __name__ == "__main__":
    main()
