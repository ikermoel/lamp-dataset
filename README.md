# LAMP Synthetic Dataset

10,000 synthetic FEVER-style examples in English, generated with **Qwen3-4B-Instruct-2507** on **AMD MI300X** via vLLM. Texts are completely fictional to avoid parametric contamination.

## Files

### Text dataset (binary classification: SUPPORTS / REFUTES)

| File | Examples | Claims |
|---|---|---|
| `data/dataset.jsonl` | 10,000 | 20,000 |
| `data/train.jsonl`   | 7,000  | 14,000 |
| `data/eval.jsonl`    | 3,000  | 6,000 |

### Last-token activations (for LAMP autoencoder training)

Last-token hidden state from each of Qwen3-4B's 36 layers, extracted on the full text of each example (max 512 input tokens, padded left).

Shape per example: `[36, 2560]` (num_layers × hidden_size). Stored as `float16`.

Files in `data/activations/`:
- `manifest.json` — model name, dimensions
- `acts_shard_NN.npz` — sharded numpy archives, ~500 examples each, with keys `activations` (`float16[N, 36, 2560]`) and `ids` (`int64[N]`)

Loading example:
```python
import glob, numpy as np
shards = sorted(glob.glob('data/activations/acts_shard_*.npz'))
acts = np.concatenate([np.load(s)['activations'] for s in shards])
ids  = np.concatenate([np.load(s)['ids']         for s in shards])
# acts.shape == (10000, 36, 2560)
```

## Generated for the LAMP project
Latent Autoencoded Memory Persistence — OpenResearch Cohort 1, May 2026.

## Phase 3 results (LoRA + memory-token classifier)

Trained LoRA adapter on Qwen3-4B-Instruct-2507 to read the AE memory token z (256d) as input embedding and predict claim labels.

| Condition | Accuracy |
|---|---|
| No context (vanilla Qwen3-4B, claim only) | 0.6325 |
| Full text (vanilla Qwen3-4B, text + claim, upper bound) | 0.9032 |
| **Memory token (LAMP — LoRA + 256d z + claim)** | **0.9035** |

**LAMP / upper bound = 1.0004** (threshold publishable 0.70, top-tier 0.85)

Compression: 36 layers x 2560 hidden = 92,160 floats -> 256 floats = **360x compression** with full-task accuracy preserved.

See `data/phase3_results.json` for raw numbers.
