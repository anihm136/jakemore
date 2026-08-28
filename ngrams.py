import os
import time
import pickle
from dataclasses import dataclass
import jax
import jax.numpy as jnp

from common import (
    VOCAB_SIZE,
    build_tokenized_dataset,
    itos,
    load_names,
)


@dataclass
class NgramConfig:
    """Configuration for the count-based n-gram language model."""

    data_path: str = "names.txt"
    save_dir: str = "models"
    ngram_size: int = 4
    smoothing: float = 1.0  # Laplace smoothing weight
    train_ratio: float = 0.8
    seed: int = 0
    num_samples: int = 10
    max_length: int = 20


def fit_ngram_counts(
    xs: jnp.ndarray, ys: jnp.ndarray, ngram_size: int, vocab_size: int
) -> jnp.ndarray:
    """Builds the (27, ... 27) count tensor from token transitions in a vectorized pass."""
    ngrams = jnp.column_stack([xs, ys])
    dims = (vocab_size,) * ngram_size
    flat_indices = jnp.ravel_multi_index(ngrams.T, dims=dims, mode="clip")
    counts_flat = jnp.bincount(flat_indices, length=vocab_size**ngram_size)
    return counts_flat.reshape(dims).astype(jnp.float32)


def evaluate_nll(
    logprobs: jnp.ndarray, xs: jnp.ndarray, ys: jnp.ndarray, ngram_size: int, vocab_size
) -> tuple[float, float]:
    """Computes Negative Log-Likelihood and Perplexity across transitions."""
    ngrams = jnp.column_stack([xs, ys])
    dims = (vocab_size,) * ngram_size
    flat_indices = jnp.ravel_multi_index(ngrams.T, dims=dims, mode="clip")
    selected_logprobs = logprobs.ravel()[flat_indices]
    mean_nll = -float(jnp.mean(selected_logprobs))
    perplexity = float(jnp.exp(mean_nll))
    return mean_nll, perplexity


def sample(
    probs: jnp.ndarray,
    key: jax.Array,
    ngram_size: int,
    num_samples: int = 10,
    max_length: int = 20,
) -> list[str]:
    """Generates names character-by-character autoregressively."""
    generated_names = []
    for _ in range(num_samples):
        cur_tokens = [0] * (ngram_size - 1)
        out_chars = []
        for _ in range(max_length):
            key, subkey = jax.random.split(key)
            context = tuple(cur_tokens[-(ngram_size - 1) :])
            next_probs = probs[context]
            next_token = int(jax.random.categorical(subkey, jnp.log(next_probs)))
            if next_token == 0:
                break
            out_chars.append(itos(next_token))
            cur_tokens.append(next_token)
        generated_names.append("".join(out_chars))
    return generated_names


def main(config: NgramConfig = NgramConfig()):
    print("=" * 60)
    print(f"  jakemore: {config.ngram_size}-gram Count Language Model (JAX)")
    print("=" * 60)

    # 1. Load dataset & build tokenized n-grams
    names = load_names(config.data_path)
    xs, ys = build_tokenized_dataset(names, ngram_size=config.ngram_size)

    # Split dataset
    split_idx = int(round(xs.shape[0] * config.train_ratio))
    train_xs, train_ys = xs[:split_idx], ys[:split_idx]
    test_xs, test_ys = xs[split_idx:], ys[split_idx:]

    print(
        f"Loaded {len(names):,} names ({xs.shape[0]:,} {config.ngram_size}-gram transitions)"
    )
    print(f"Train set: {train_xs.shape[0]:,} transitions")
    print(f"Test set:  {test_xs.shape[0]:,} transitions")

    # 2. Fit counts & apply Laplace smoothing
    start = time.perf_counter()
    counts = fit_ngram_counts(train_xs, train_ys, config.ngram_size, VOCAB_SIZE)
    smoothed_counts = counts + config.smoothing
    probs = smoothed_counts / smoothed_counts.sum(axis=-1, keepdims=True)
    logprobs = jnp.log(probs)
    stop = time.perf_counter()
    train_time = stop - start

    # 3. Evaluate NLL & Perplexity
    train_nll, train_perplexity = evaluate_nll(
        logprobs, train_xs, train_ys, config.ngram_size, VOCAB_SIZE
    )
    test_nll, test_perplexity = evaluate_nll(
        logprobs, test_xs, test_ys, config.ngram_size, VOCAB_SIZE
    )

    print(f"\nLaplace Smoothing:  {config.smoothing}")
    print(f"Fit Time:           {train_time * 1000:.2f} ms")
    print(
        f"Train NLL:          {train_nll:.4f} | Train Perplexity: {train_perplexity:.4f}"
    )
    print(
        f"Test NLL:           {test_nll:.4f} | Test Perplexity:  {test_perplexity:.4f}"
    )

    # 4. Generate sample names
    master_key = jax.random.key(config.seed)
    generated = sample(
        probs,
        master_key,
        ngram_size=config.ngram_size,
        num_samples=config.num_samples,
        max_length=config.max_length,
    )

    print("\nGenerated Names:")
    for idx, name in enumerate(generated, start=1):
        print(f"  {idx:2d}. {name}")

    # 5. Save model
    os.makedirs(config.save_dir, exist_ok=True)
    save_path = os.path.join(config.save_dir, "ngram_model.pkl")
    model_data = {
        "counts": counts,
        "probs": probs,
        "logprobs": logprobs,
        "smoothing": config.smoothing,
        "ngram_size": config.ngram_size,
        "train_nll": train_nll,
        "train_perplexity": train_perplexity,
        "test_nll": test_nll,
        "test_perplexity": test_perplexity,
    }
    with open(save_path, "wb") as f:
        pickle.dump(model_data, f)
    print(f"\nModel saved to '{save_path}'\n")


if __name__ == "__main__":
    main()
