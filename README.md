# LAMP Synthetic Dataset

10,000 synthetic FEVER-style examples in English, generated with **Qwen3.5-4B** on **AMD MI300X**, using completely fictional content to avoid parametric contamination.

| File | Examples |
|---|---|
| `data/dataset.jsonl` | 10,000 |
| `data/train.jsonl`   | 7,000 |
| `data/eval.jsonl`    | 3,000 |

Each example:
```json
{
  "id": 0,
  "text": "...",
  "claims": [
    {"claim": "...", "label": "SUPPORTS"},
    {"claim": "...", "label": "REFUTES"},
    {"claim": "...", "label": "NOT ENOUGH INFO"}
  ]
}
```

Generated for the LAMP project (Latent Autoencoded Memory Persistence) — OpenResearch Cohort 1, May 2026.
