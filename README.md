# LAMP Synthetic Dataset (binary)

10,000 synthetic FEVER-style examples in English, generated with **Qwen3-4B-Instruct-2507** on **AMD MI300X** via vLLM, on completely fictional content to avoid parametric contamination.

**Note:** the original 3-class FEVER protocol included a NOT ENOUGH INFO label, but those claims had ~50% label noise (model often produced REFUTES disguised as NEI). To keep label quality high, all NEI claims have been removed. Each example now has 2 claims: one SUPPORTS and one REFUTES.

| File | Examples | Claims per example | Total claims |
|---|---|---|---|
| `data/dataset.jsonl` | 10,000 | 2 (SUPPORTS, REFUTES) | 20,000 |
| `data/train.jsonl`   | 7,000  | 2 | 14,000 |
| `data/eval.jsonl`    | 3,000  | 2 | 6,000 |

Each example:
```json
{
  "id": 0,
  "text": "...",
  "claims": [
    {"claim": "...", "label": "SUPPORTS"},
    {"claim": "...", "label": "REFUTES"}
  ]
}
```

Generated for the LAMP project (Latent Autoencoded Memory Persistence) — OpenResearch Cohort 1, May 2026.
