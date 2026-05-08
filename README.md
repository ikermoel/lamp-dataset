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
