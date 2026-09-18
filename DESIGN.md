# Model Tuning Design

This workstream trains a task-specific logistic reranker over BM25 retrieval candidates. Features are BM25 score, query/document overlap, and query/title overlap. Training uses labeled retrieval pairs; validation is separate. This is genuine parameter training, but it is **not LLM weight fine-tuning**.
