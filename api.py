"""API de Vigie : expose les lots, les alertes, les diagnostics de l'agent et l'évaluation.

Lancer depuis la racine du projet (les chemins data/ sont relatifs) :
    uvicorn api:app --reload
puis ouvrir http://127.0.0.1:8000
"""

import json
import re
import time
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from langchain_core.runnables import RunnableConfig
from langfuse import Langfuse, get_client
from langfuse.langchain import CallbackHandler

from agent import graph
from eval import mesurer
from investigate import construire_message
from tools import compute_psi

DATA = Path("data")
DIAGNOSTICS = DATA / "diagnostics"
DASHBOARD = Path(__file__).parent / "static" / "index.html"
FORMAT_JOUR = re.compile(r"^day_\d{2}$")

app = FastAPI(title="Vigie", description="Agent d'astreinte pour un modèle de crédit en production.")


def _lire_json(chemin: Path) -> dict:
    if not chemin.exists():
        return {}
    with open(chemin, encoding="utf-8") as f:
        return json.load(f)


def _jours() -> list[str]:
    return sorted(p.stem for p in (DATA / "batches").glob("day_*.csv"))


def _verifier_jour(day: str) -> None:
    if not FORMAT_JOUR.match(day) or day not in _jours():
        raise HTTPException(404, f"Lot inconnu : {day}. Lots disponibles : {', '.join(_jours())}")


def _diagnostic(day: str) -> dict | None:
    """Dernier diagnostic connu : celui lancé depuis l'API, sinon celui de la dernière évaluation."""
    enregistre = _lire_json(DIAGNOSTICS / f"{day}.json")
    if enregistre:
        return enregistre

    for enquete in _lire_json(DATA / "eval.json").get("enquetes", []):
        if enquete.get("jour") == day and "erreur" not in enquete:
            return {
                "jour": day,
                "source": "eval",
                "cause": enquete["cause_obtenue"],
                "variables": enquete["variables_obtenues"],
                "gravite": enquete["gravite"],
                "action": enquete["action_obtenue"],
                "preuves": enquete["preuves"],
                "appels_outils": enquete["appels_outils"],
                "duree_s": enquete["duree_s"],
                "cout_usd": enquete["cout_usd"],
                "outils": None,
                "trace_url": None,
            }
    return None


@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(DASHBOARD)


@app.get("/api/lots")
def lister_lots() -> list[dict]:
    """Tous les lots, avec leur alerte et le dernier diagnostic de l'agent."""
    alertes = _lire_json(DATA / "alerts.json")
    return [
        {"jour": day, "alerte": alertes.get(day), "diagnostic": _diagnostic(day)}
        for day in _jours()
    ]


@app.get("/api/lots/{day}")
def detail_lot(day: str) -> dict:
    """Un lot : son alerte, le PSI de chaque variable et le dernier diagnostic."""
    _verifier_jour(day)
    return {
        "jour": day,
        "alerte": _lire_json(DATA / "alerts.json").get(day),
        "psi": compute_psi.invoke({"day": day})["psi"],
        "diagnostic": _diagnostic(day),
    }


@app.post("/api/lots/{day}/enquete")
def lancer_enquete(day: str) -> dict:
    """Lance l'agent sur l'alerte du jour. Appel payant au LLM : 10 à 30 secondes."""
    _verifier_jour(day)
    alerte = _lire_json(DATA / "alerts.json").get(day)
    if alerte is None:
        raise HTTPException(409, f"Aucune alerte pour {day} : pas d'enquête à lancer.")

    # Un identifiant de trace choisi d'avance permet de renvoyer le lien Langfuse.
    trace_id = Langfuse.create_trace_id()
    config: RunnableConfig = {
        "callbacks": [CallbackHandler(trace_context={"trace_id": trace_id})],
        "metadata": {"langfuse_tags": ["api", day]},
    }

    debut = time.perf_counter()
    try:
        resultat = graph.invoke(
            {"messages": [{"role": "user", "content": construire_message(day, alerte)}]},
            config=config,
        )
    except Exception as erreur:
        raise HTTPException(502, f"L'enquête a échoué : {erreur}")
    finally:
        get_client().flush()
    duree = time.perf_counter() - debut

    diagnostic = resultat["structured_response"]
    outils = [
        appel["name"]
        for message in resultat["messages"]
        for appel in getattr(message, "tool_calls", None) or []
    ]

    enregistrement = {
        "jour": day,
        "source": "api",
        "date": datetime.now().isoformat(timespec="seconds"),
        **diagnostic.model_dump(),
        "outils": outils,
        "duree_s": round(duree, 1),
        **mesurer(resultat),
        "trace_url": get_client().get_trace_url(trace_id=trace_id),
    }

    DIAGNOSTICS.mkdir(parents=True, exist_ok=True)
    with open(DIAGNOSTICS / f"{day}.json", "w", encoding="utf-8") as f:
        json.dump(enregistrement, f, indent=4, ensure_ascii=False)

    return enregistrement


@app.get("/api/eval")
def evaluation() -> dict:
    """Résultat de la dernière évaluation (python eval.py)."""
    resultat = _lire_json(DATA / "eval.json")
    if not resultat:
        raise HTTPException(404, "Aucune évaluation : lancer d'abord python eval.py")
    return resultat
