# jakemore

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/framework-JAX-red.svg)](https://github.com/google/jax)

A clean, minimal, and performant **JAX** implementation of Andrej Karpathy's [makemore](https://github.com/karpathy/makemore).

`jakemore` implements autoregressive character-level language modeling from first principles using pure JAX and Google Grain.

---

## Highlights

- **Pure JAX**: Fast JIT compilation (`jax.jit`), vectorized matrix operations, batch transformations (`jax.vmap`), and functional PRNG keys (`jax.random.split`).
- **N-gram Count Model**: Vectorized multidimensional statistical counting with Laplace smoothing.
- **Bengio et al. (2003) MLP**: Character embedding lookup table, hidden layer with tanh activations, output linear layer, and step learning rate decay.
- **Grain Data Pipeline**: Mini-batch iteration with reproducible shuffling, streaming and looping via `grain.python`.
- **Clean Modular Design**: Shared tokenization and dataset primitives in `common.py`.

---

## Quickstart

### 1. Run the N-gram Count Model

Fits the empirical $N$-gram transition tensor with Laplace smoothing, reports train/test NLL and perplexity, samples names, and saves the model to `models/ngram_model.pkl`:

```bash
uv run python ngrams.py
```

### 2. Run the MLP Neural Network

Trains the character embedding MLP using JIT-compiled SGD with Grain data loading, reports training loss, evaluates train/test splits, samples names, and saves weights to `models/nn_model.pkl`:

```bash
uv run python nn.py
```

### 3. Run Unit Tests

```bash
uv run pytest -v
```

---

## Configuration

Each script exposes a simple `@dataclass` at the top of the file:

- **`NgramConfig` (`ngrams.py`)**: `data_path`, `save_dir`, `ngram_size`, `smoothing`, `train_ratio`, `seed`, `num_samples`, `max_length`.
- **`NNConfig` (`nn.py`)**: `data_path`, `save_dir`, `ngram_size`, `emb_size`, `hidden_dim`, `learning_rate`, `lr_decay_step`, `lr_decay_factor`, `num_steps`, `batch_size`, `reg_weight`, `train_ratio`, `seed`, `num_samples`, `max_length`, `temperature`, `log_interval`.
