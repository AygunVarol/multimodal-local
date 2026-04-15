
import os
from rdflib import Graph, Namespace
from rdflib.namespace import RDF, RDFS
from typing import Dict, Any, List

DATA_KG = os.getenv("KG_PATH", "../data/kg.ttl")
EX = Namespace("http://example.org/")

class KGStore:
    def __init__(self, path: str = DATA_KG):
        self.g = Graph()
        self.g.parse(path, format="turtle")
        self.path = path

    def sparql(self, query: str) -> List[Dict[str, Any]]:
        results = []
        for row in self.g.query(query):
            results.append({var: str(val) for var,val in row.asdict().items()})
        return results

    def recommended_range(self, metric_label: str):
        metric = None
        wanted = metric_label.lower()
        for subject in self.g.subjects(RDF.type, EX.Metric):
            label = self.g.value(subject, RDFS.label)
            if label is not None and str(label).lower() == wanted:
                metric = subject
                break
        if metric is None:
            return None
        out = {}
        rec_min = self.g.value(metric, EX.recommendedMin)
        rec_max = self.g.value(metric, EX.recommendedMax)
        if rec_min is not None:
            try: out["min"] = float(rec_min)
            except: pass
        if rec_max is not None:
            try: out["max"] = float(rec_max)
            except: pass
        return out if out else None

    def short_context_for(self, text: str, limit: int = 6) -> str:
        words = [w.lower() for w in text.split() if len(w) >= 3]
        triples = []
        for s, p, o in self.g.triples((None, RDFS.label, None)):
            if any(w in str(o).lower() for w in words):
                subject_name = s.split("/")[-1]
                triples.append(f"{subject_name} --label--> {o}")
                if (s, RDF.type, EX.Metric) in self.g:
                    rec_min = self.g.value(s, EX.recommendedMin)
                    rec_max = self.g.value(s, EX.recommendedMax)
                    if rec_min is not None or rec_max is not None:
                        min_text = str(rec_min) if rec_min is not None else "-inf"
                        max_text = str(rec_max) if rec_max is not None else "inf"
                        triples.append(f"{subject_name} --recommendedRange--> {min_text} to {max_text}")
                if len(triples) >= limit: break
        for policy in self.g.subjects(RDF.type, EX.Policy):
            label = self.g.value(policy, RDFS.label)
            if label is None:
                continue
            lab = str(label)
            if any(w in lab.lower() for w in words):
                triples.append(f"Policy: {lab}")
                if len(triples) >= limit: break
        return "\n".join(triples)
