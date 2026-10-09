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

## Performance Microbenchmarks

Summary of microbenchmarks from [`jakemore_performance_comparisons.ipynb`](jakemore_performance_comparisons.ipynb) evaluating a 2-layer MLP with BatchNorm (4,137 parameters, batch size 32, context length 3) on CPU:

| Workload | PyTorch Eager | PyTorch (`torch.compile`) | JAX Eager | JAX (`jax.jit`) |
| :--- | :--- | :--- | :--- | :--- |
| **Forward Pass** | 456 µs ± 110 µs | 383 µs ± 135 µs | 4.27 ms ± 1.06 ms | **80.5 µs ± 2.8 µs** |
| **Manual Backward** | 607 µs ± 99.3 µs | **183 µs ± 24.8 µs** | 6.98 ms ± 65.4 µs | 295 µs ± 59.6 µs |
| **Forward + Backward (Autodiff)** | 718 µs ± 140 µs | 737 µs ± 111 µs | 33.1 ms ± 935 µs | **310 µs ± 11.1 µs** |

- **Eager execution**: PyTorch has lower CPU dispatch overhead per op than JAX in eager mode across all workloads.
- **JIT compilation**: `jax.jit` fuses operations and eliminates dispatch overhead, running 5.7x faster than PyTorch eager on forward-only (80.5 µs vs. 456 µs) and 2.3x faster on combined forward + backward autodiff (310 µs vs. 718 µs).
- **Manual backprop vs. Autodiff**: `torch.compile` is fastest on the standalone manual backward pass (183 µs), while `jax.jit` is fastest on the combined forward + backward pass using autodiff (`jax.value_and_grad`).

---

## Configuration

Each script exposes a simple `@dataclass` at the top of the file:

- **`NgramConfig` (`ngrams.py`)**: `data_path`, `save_dir`, `ngram_size`, `smoothing`, `train_ratio`, `seed`, `num_samples`, `max_length`.
- **`NNConfig` (`nn.py`)**: `data_path`, `save_dir`, `ngram_size`, `emb_size`, `hidden_dim`, `learning_rate`, `lr_decay_step`, `lr_decay_factor`, `num_steps`, `batch_size`, `reg_weight`, `train_ratio`, `seed`, `num_samples`, `max_length`, `temperature`, `log_interval`.
