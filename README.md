## NALLAR

Web interface: https://nallar.or.id

Public engineering evidence for this repository is documented in the
tests, evaluation artifacts, and evidence files included here.
💻 Public AI engineering portfolio: https://github.com/andreacentauri-sketch

# Model Tuning

> Recruiter-facing public presentation generated from the verified local NALLAR portfolio package.

## 30-second summary

This project is presented around four questions: **what was built, how it was tested, what was measured, and what is not being claimed**.

## Evidence model

- Runnable implementation or portfolio artifact.
- Executable tests and/or quantitative evaluation.
- Reproducible local commands.
- Explicit limitations.

## Proof points

- Real task-specific parameter training using a logistic reranker.
- **96 labeled training examples.**
- Training loss: **1.040 → 0.103**.
- Validation top-1: baseline **100.0%**, tuned **100.0%**.
- **12/12 tests passed.**

## Evidence boundary

This is task-specific model training. **LLM weight fine-tuning was not performed.**

---

## Reproduce locally

See the project files and original run notes below. Use the repository's own test/evaluation commands and inspect the generated evidence artifacts rather than relying on screenshots alone.


### Original run notes

# NALLAR Model Tuning

Train: `python tuner.py --data data/tuning_data.json --corpus ../03_RAG_TELEMETRY_EVALS/data/sample_corpus.json --model-out MODEL.json --metrics-out METRICS.json`

Test: `python -m unittest discover -s tests -v`


## Public claim boundary

This repository is published as an evidence-backed portfolio artifact. Test and evaluation results apply to the documented local/bundled scope. No production deployment, foundation-model training, or other unverified capability is implied.
