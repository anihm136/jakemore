# jakemore

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/framework-JAX-red.svg)](https://github.com/google/jax)

A clean, minimal, and performant **JAX** implementation of Andrej Karpathy's [makemore](https://github.com/karpathy/makemore)

`jakemore` implements autoregressive character-level language modeling from first principles using pure JAX

---

## Highlights

- **Pure JAX**: Fast JIT compilation (`jax.jit`), transparent matrix operations, and functional PRNG keys (`jax.random.split`). Implementation is tuned to minimize JIT recompiles
- **Statistical & Neural Duality**: Empirically demonstrates how a single-layer neural network minimizing cross-entropy loss with weight decay converges to smoothed empirical bigram counts.

---

## Differences from Original `makemore`

| Dimension          | Original `makemore` (Part 1)               | `jakemore` (JAX Port)                                                                                       |
| :----------------- | :----------------------------------------- | :---------------------------------------------------------------------------------------------------------- |
| **Framework**      | PyTorch (`torch.Tensor`, `torch.autograd`) | JAX (`jax.numpy`, `jax.jit`, `jax.value_and_grad`)                                                          |
| **Token Pipeline** | Iterative per-word pair loops in Python    | Continuous 1D token stream with batch tensor shaping (to maintain static batch shapes and avoid recompiles) |
| **Configuration**  | Inline script variables & magic numbers    | Typed `@dataclass` configs (`BigramConfig`, `NNConfig`)                                                     |
| **Persistence**    | Ephemeral in-memory execution              | Pickled model serialization (`models/`) & unified `main.py` loader                                          |

---

## Quickstart

### 1. Run the Bigram Count Model

Fits the empirical transition matrix with Laplace smoothing, reports NLL, samples 10 names, and saves the model to `models/bigram_model.pkl`:

```bash
python bigrams.py
# or: uv run python bigrams.py
```

### 2. Run the Single-Layer Neural Network

Trains a single linear layer using JIT-compiled gradient descent, reports training progress, samples 10 names, and saves weights to `models/nn_model.pkl`:

```bash
python nn.py
# or: uv run python nn.py
```

### 3. Evaluate & Compare Saved Models

Loads both saved models from `models/`, compares their loss/perplexity on `names.txt`, and samples 5 random names from each:

```bash
python main.py
# or: uv run python main.py
```

---

## Empirical Benchmarks

Trained on 32,033 names (`names.txt`, 228,146 bigram transitions):

| Model              | Optimization               | Regularization / Smoothing | Negative Log-Likelihood | Perplexity  |
| :----------------- | :------------------------- | :------------------------- | :---------------------: | :---------: |
| **Count Model**    | Analytic MLE               | Laplace ($k = 1.0$)        |       **2.4546**        | **11.6415** |
| **Neural Network** | 100 steps GD ($\eta = 10$) | L2 ($\lambda = 0.01$)      |       **2.4636**        | **11.7471** |

---

## Configuration

Each script exposes a simple `@dataclass` at the top of the file:

- **`BigramConfig` (`bigrams.py`)**: `data_path`, `save_dir`, `smoothing`, `seed`, `num_samples`, `max_length`.
- **`NNConfig` (`nn.py`)**: `data_path`, `save_dir`, `learning_rate`, `num_steps`, `batch_size`, `reg_weight`, `seed`, `num_samples`, `temperature`, `max_length`, `log_interval`.
