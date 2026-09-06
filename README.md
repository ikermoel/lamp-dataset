# LAMP Synthetic Dataset

LAMP is a synthetic, English-language dataset with 10,000 fictional documents and 20,000 FEVER-style claims for binary fact verification. Each document is paired with one supported claim and one refuted claim. A separate open-answer question-answering version contains **30,000 question-answer pairs across the same 10,000 documents**, with three questions per document.

The dataset was generated with **Qwen3-4B-Instruct-2507** on **AMD MI300X** via vLLM. All people, organizations, products, and events described in the texts are fictional to reduce the risk of parametric contamination.

## Dataset contents

| Path | Contents |
|---|---|
| `data/dataset.jsonl` | Full dataset: 10,000 documents and 20,000 claims |
| `data/train.jsonl` | Training split: 7,000 documents and 14,000 claims |
| `data/eval.jsonl` | Evaluation split: 3,000 documents and 6,000 claims |
| `data/qa_dataset.jsonl` | 10,000 documents with 30,000 open-answer question-answer pairs (3 per document) |
| `data/ood_examples.jsonl` | Additional out-of-distribution examples |
| `data/activations/` | Last-token hidden-state activations for all 10,000 documents |

## Format

The main files use JSON Lines. Each row contains a fictional source text, two claims, their labels, and a numeric ID.

| Field | Type | Description |
|---|---|---|
| `id` | integer | Unique example identifier |
| `text` | string | Fictional source document |
| `claims` | array | Claims paired with the source document |
| `claims[].claim` | string | Claim to verify |
| `claims[].label` | string | `SUPPORTS` or `REFUTES` |

## Mini sample

The text below is shortened for readability; the claims and label structure match the dataset.

```json
{
  "id": 0,
  "text": "In 2023, Prague-based telecommunications firm Veridium Commu was founded with a focus on rural connectivity...",
  "claims": [
    {
      "claim": "Veridium Commu launched the 'NexLink' initiative in 2024 with an annual budget of €2.8 million.",
      "label": "SUPPORTS"
    },
    {
      "claim": "Veridium Commu launched the 'NexLink' initiative in 2025 with an annual budget of €3.5 million.",
      "label": "REFUTES"
    }
  ]
}
```

## Open-answer question answering

`data/qa_dataset.jsonl` contains **10,000 JSONL rows and 30,000 question-answer pairs**. Each row groups three questions and their reference answers with one source document. These are open-answer questions, rather than `SUPPORTS` / `REFUTES` labels.

| Field | Type | Description |
|---|---|---|
| `id` | integer | Document identifier |
| `text` | string | Fictional source document |
| `qas` | array | Three question-answer pairs for the document |
| `qas[].q` | string | Question about the document |
| `qas[].a` | string | Reference answer |

The following example preserves the actual questions and answers; the source text is shortened for readability.

```json
{
  "id": 0,
  "text": "In 2023, Prague-based telecommunications firm Veridium Commu was founded with a focus on rural connectivity...",
  "qas": [
    {
      "q": "When was Veridium Commu founded?",
      "a": "2023"
    },
    {
      "q": "How many customers does Veridium serve?",
      "a": "420000"
    },
    {
      "q": "What is the average latency?",
      "a": "28 milliseconds"
    }
  ]
}
```

This version can support document comprehension and memory experiments: provide the source document as context, then evaluate answers with or without that document available. It complements rule-learning datasets by testing retention of new facts; it does not itself establish generalization to new rule families. To avoid document leakage, keep all questions from the same document in the same train or evaluation partition. The text and QA files can be used independently of the stored activations.

## Activations

`data/activations/` contains the last-token hidden state from each of the model's 36 layers, extracted from the full text of every example with a maximum input length of 512 tokens and left padding.

- Shape per example: `[36, 2560]` (`num_layers × hidden_size`)
- Data type: `float16`
- `manifest.json`: model and tensor dimensions
- `acts_shard_NN.npz`: approximately 500 examples per shard, with `activations` and `ids` arrays

```python
import glob
import numpy as np

shards = sorted(glob.glob("data/activations/acts_shard_*.npz"))
activations = np.concatenate([np.load(shard)["activations"] for shard in shards])
ids = np.concatenate([np.load(shard)["ids"] for shard in shards])

# activations.shape == (10000, 36, 2560)
```

## Project

Generated for a future research project: **LAMP (Latent Autoencoded Memory Persistence)**.
