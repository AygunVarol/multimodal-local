
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
from ..services.kg_store import KGStore
from ..services.ollama_client import generate

router = APIRouter()
kg = KGStore()

class SPARQLRequest(BaseModel):
    query: str

class AskKGRequest(BaseModel):
    question: str
    model: Optional[str] = None

@router.post("/kg/sparql")
def kg_sparql(req: SPARQLRequest):
    try:
        rows = kg.sparql(req.query)
        return {"rows": rows}
    except Exception as e:
        return {"error": str(e)}

@router.post("/kg/ask")
def kg_ask(req: AskKGRequest):
    system = (
        "Translate a natural language question about locations, displays, metrics, datasets, or policies "
        "into a SPARQL SELECT query over the http://example.org/ namespace. Use rdfs:label lookups. "
        "Return ONLY the SPARQL string."
    )
    prompt = f"""{system}

Question: {req.question}

PREFIX ex: <http://example.org/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

Output: SPARQL SELECT query only.
"""
    sparql = ""
    try:
        gen = generate(prompt, model=req.model)
        sparql = (gen.get("content") or "").strip()
    except Exception:
        pass
    if not sparql:
        sparql = """
PREFIX ex: <http://example.org/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?thing ?label WHERE { ?thing rdfs:label ?label . } LIMIT 20
"""
    try:
        rows = kg.sparql(sparql)
        return {"sparql": sparql, "rows": rows}
    except Exception as e:
        return {"sparql": sparql, "error": str(e)}
