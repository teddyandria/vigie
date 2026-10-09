"""Lance l'agent sur chaque alerte et compare son diagnostic à data/incidents.json."""

import json
import time
from datetime import datetime

from langfuse import get_client
from langchain_core.runnables import RunnableConfig
from langfuse.langchain import CallbackHandler

from agent import graph
from investigate import construire_message
from tools import TOOLS

# Tarif gpt-5.5 en dollars par million de tokens (sans remise de cache ni batch).
PRIX_ENTREE_PAR_MILLION = 5.0
PRIX_SORTIE_PAR_MILLION = 30.0

# Exclut l'appel de sortie structurée, qui n'est pas un outil d'enquête.
NOMS_OUTILS = {outil.name for outil in TOOLS}


def mesurer(resultat: dict) -> dict:
    """Compte les appels d'outils et les tokens d'une enquête terminée."""
    appels = 0
    tokens_entree = 0
    tokens_sortie = 0

    for message in resultat["messages"]:
        for appel in getattr(message, "tool_calls", None) or []:
            if appel["name"] in NOMS_OUTILS:
                appels += 1

        usage = getattr(message, "usage_metadata", None)
        if usage:
            tokens_entree += usage.get("input_tokens", 0)
            tokens_sortie += usage.get("output_tokens", 0)

    cout = None
    if PRIX_ENTREE_PAR_MILLION is not None and PRIX_SORTIE_PAR_MILLION is not None:
        cout = (tokens_entree * PRIX_ENTREE_PAR_MILLION + tokens_sortie * PRIX_SORTIE_PAR_MILLION) / 1_000_000

    return {
        "appels_outils": appels,
        "tokens_entree": tokens_entree,
        "tokens_sortie": tokens_sortie,
        "cout_usd": round(cout, 4) if cout is not None else None,
    }


def evaluer_jour(day: str, alerte: dict, verite: dict, session_id: str) -> dict:
    """Lance une enquête sur un jour et la compare à la vérité."""
    handler = CallbackHandler()
    config: RunnableConfig = {
        "callbacks": [handler],
        "metadata": {"langfuse_session_id": session_id, "langfuse_tags": ["eval", day]},
    }

    debut = time.perf_counter()
    try:
        resultat = graph.invoke(
            {"messages": [{"role": "user", "content": construire_message(day, alerte)}]},
            config=config,
        )
    except Exception as erreur:
        return {
            "jour": day,
            "erreur": str(erreur),
            "duree_s": round(time.perf_counter() - debut, 1),
            "cause_ok": False,
            "variables_ok": False,
            "action_ok": False,
        }
    duree = time.perf_counter() - debut

    diagnostic = resultat["structured_response"]

    return {
        "jour": day,
        "cause_attendue": verite["cause"],
        "cause_obtenue": diagnostic.cause,
        "cause_ok": diagnostic.cause == verite["cause"],
        "variables_attendues": verite["variables"],
        "variables_obtenues": diagnostic.variables,
        "variables_ok": set(diagnostic.variables) == set(verite["variables"]),
        "action_attendue": verite["action"],
        "action_obtenue": diagnostic.action,
        "action_ok": diagnostic.action == verite["action"],
        "gravite": diagnostic.gravite,
        "preuves": diagnostic.preuves,
        "duree_s": round(duree, 1),
        **mesurer(resultat),
    }


def afficher(lignes: list, scores: dict) -> None:
    """Affiche le tableau des résultats et les scores."""
    def ok(valeur):
        return "oui" if valeur else "NON"

    entete = f"{'jour':<8} {'cause obtenue':<20} {'cause':<6} {'var.':<6} {'action':<7} {'outils':>6} {'durée':>7} {'coût':>8}"
    print(entete)
    print("-" * len(entete))
    for l in lignes:
        if "erreur" in l:
            print(f"{l['jour']:<8} ERREUR : {l['erreur'][:60]}")
            continue
        cout = f"{l['cout_usd']:.4f}$" if l["cout_usd"] is not None else "n/d"
        print(
            f"{l['jour']:<8} {l['cause_obtenue']:<20} {ok(l['cause_ok']):<6} {ok(l['variables_ok']):<6} "
            f"{ok(l['action_ok']):<7} {l['appels_outils']:>6} {l['duree_s']:>6}s {cout:>8}"
        )

    n = scores["nb_enquetes"]
    print()
    print(f"Bonne cause      : {scores['cause_ok']} cas sur {n}")
    print(f"Bonnes variables : {scores['variables_ok']} cas sur {n}")
    print(f"Bonne action     : {scores['action_ok']} cas sur {n}")


def main():
    with open("data/alerts.json", encoding="utf-8") as f:
        alertes = json.load(f)
    with open("data/incidents.json", encoding="utf-8") as f:
        incidents = json.load(f)

    session_id = f"eval-{datetime.now():%Y%m%d-%H%M%S}"

    lignes = []
    try:
        for day, alerte in alertes.items():
            if day not in incidents:
                print(f"{day} : alerte sans vérité dans incidents.json, ignoré")
                continue
            print(f"Enquête sur {day}...")
            lignes.append(evaluer_jour(day, alerte, incidents[day], session_id))
    finally:
        get_client().flush()

    scores = {
        "nb_enquetes": len(lignes),
        "cause_ok": sum(l["cause_ok"] for l in lignes),
        "variables_ok": sum(l["variables_ok"] for l in lignes),
        "action_ok": sum(l["action_ok"] for l in lignes),
    }

    print()
    afficher(lignes, scores)

    with open("data/eval.json", "w", encoding="utf-8") as f:
        json.dump(
            {"session": session_id, "scores": scores, "enquetes": lignes},
            f,
            indent=4,
            ensure_ascii=False,
        )
    print(f"\nRésultats enregistrés dans data/eval.json (session Langfuse : {session_id})")


if __name__ == "__main__":
    main()
