#!/usr/bin/env python3
from __future__ import annotations
import argparse, importlib.util, json, math, os, re
from pathlib import Path
from typing import Any, Dict, List, Tuple

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")

def tokenize(text: str) -> List[str]:
    return [x.lower() for x in TOKEN_RE.findall(text)]

def rag_root() -> Path:
    raw = os.environ.get("NALLAR_RAG_PACKAGE")
    return Path(raw).expanduser().resolve() if raw else (Path(__file__).resolve().parents[1] / "03_RAG_TELEMETRY_EVALS").resolve()

def load_rag():
    path = rag_root() / "rag_engine.py"
    spec = importlib.util.spec_from_file_location("nallar_g4_rag", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load G2 RAG module")
    mod = importlib.util.module_from_spec(spec)
    __import__("sys").modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod

FEATURE_NAMES = ["bias", "bm25_score", "query_doc_overlap", "query_title_overlap"]

def sigmoid(z: float) -> float:
    if z > 40: return 1.0
    if z < -40: return 0.0
    return 1.0 / (1.0 + math.exp(-z))

def features(query: str, result: Dict[str, Any]) -> List[float]:
    qt = set(tokenize(query))
    dt = set(tokenize(result["text"]))
    tt = set(tokenize(result["title"]))
    den = max(1, len(qt))
    return [
        1.0,
        float(result["score"]),
        len(qt & dt) / den,
        len(qt & tt) / den,
    ]

def dot(w: List[float], x: List[float]) -> float:
    return sum(a*b for a,b in zip(w,x))

def build_examples(service: Any, rows: List[Dict[str, Any]]) -> Tuple[List[List[float]], List[int]]:
    X, y = [], []
    k = max(8, len(service.corpus))
    for row in rows:
        results = service.retrieve(row["query"], k=k)
        best_by_doc = {}
        for r in results:
            if r["doc_id"] not in best_by_doc:
                best_by_doc[r["doc_id"]] = r
        for doc_id, r in best_by_doc.items():
            X.append(features(row["query"], r))
            y.append(1 if doc_id == row["expected_doc_id"] else 0)
    return X, y

def weighted_loss(weights: List[float], X: List[List[float]], y: List[int], pos_weight: float = 5.0) -> float:
    total = 0.0
    for x, yy in zip(X, y):
        p = min(max(sigmoid(dot(weights, x)), 1e-9), 1.0 - 1e-9)
        sw = pos_weight if yy else 1.0
        total += sw * (-yy * math.log(p) - (1-yy) * math.log(1-p))
    return total / max(1, len(X))

def train(service: Any, rows: List[Dict[str, Any]], epochs: int = 800, lr: float = 0.05, l2: float = 0.001) -> Dict[str, Any]:
    X, y = build_examples(service, rows)
    w = [0.0] * len(FEATURE_NAMES)
    initial = weighted_loss(w, X, y)
    for _ in range(epochs):
        grad = [0.0] * len(w)
        for x, yy in zip(X, y):
            p = sigmoid(dot(w, x))
            sw = 5.0 if yy else 1.0
            for j in range(len(w)):
                grad[j] += sw * (p - yy) * x[j]
        for j in range(len(w)):
            grad[j] /= max(1, len(X))
            if j > 0:
                grad[j] += l2 * w[j]
            w[j] -= lr * grad[j]
    final = weighted_loss(w, X, y)
    return {
        "model_type": "task_specific_logistic_reranker",
        "feature_names": FEATURE_NAMES,
        "weights": w,
        "epochs": epochs,
        "learning_rate": lr,
        "l2": l2,
        "training_examples": len(X),
        "initial_loss": initial,
        "final_loss": final,
        "loss_reduction": initial - final,
        "llm_weight_fine_tuning": False,
    }

def baseline_rank(service: Any, query: str, k: int = 8) -> List[Dict[str, Any]]:
    return service.retrieve(query, k=k)

def tuned_rank(service: Any, model: Dict[str, Any], query: str, k: int = 8) -> List[Dict[str, Any]]:
    rows = baseline_rank(service, query, k=max(k, len(service.corpus)))
    scored = []
    for r in rows:
        p = sigmoid(dot(model["weights"], features(query, r)))
        rr = dict(r)
        rr["tuned_score"] = p
        scored.append(rr)
    scored.sort(key=lambda x: (-x["tuned_score"], -x["score"], x["doc_id"], x["chunk_id"]))
    for i, row in enumerate(scored[:k], 1):
        row["rank"] = i
    return scored[:k]

def evaluate(service: Any, model: Dict[str, Any], rows: List[Dict[str, Any]], k: int = 8) -> Dict[str, Any]:
    def metrics(tuned: bool):
        hit1, rr_sum = 0, 0.0
        cases = []
        for row in rows:
            ranked = tuned_rank(service, model, row["query"], k) if tuned else baseline_rank(service, row["query"], k)
            doc_ids = []
            for r in ranked:
                if r["doc_id"] not in doc_ids:
                    doc_ids.append(r["doc_id"])
            rank = (doc_ids.index(row["expected_doc_id"]) + 1) if row["expected_doc_id"] in doc_ids else None
            hit1 += int(rank == 1)
            rr_sum += 1.0/rank if rank else 0.0
            cases.append({"id": row["id"], "expected_doc_id": row["expected_doc_id"], "rank": rank, "top_doc_id": doc_ids[0] if doc_ids else None})
        n = len(rows)
        return {"case_count": n, "top1_accuracy": hit1/n, "mrr": rr_sum/n, "cases": cases}
    return {"baseline": metrics(False), "tuned": metrics(True)}

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--model-out", required=True)
    ap.add_argument("--metrics-out", required=True)
    args = ap.parse_args()
    data = json.loads(Path(args.data).read_text(encoding="utf-8"))
    rag = load_rag()
    svc = rag.RAGService(Path(args.corpus))
    model = train(svc, data["train"])
    metrics = evaluate(svc, model, data["validation"])
    Path(args.model_out).write_text(json.dumps(model, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    Path(args.metrics_out).write_text(json.dumps(metrics, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"model": model, "metrics": metrics}, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
