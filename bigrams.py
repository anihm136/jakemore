import os
import pickle
from collections import Counter
from dataclasses import dataclass
import jax
import jax.numpy as jnp


@dataclass
class BigramConfig:
    """Configuration for the count-based bigram language model."""

    data_path: str = "names.txt"
    save_dir: str = "models"
    smoothing: float = 1.0  # Laplace smoothing (fake counts added per cell)
    seed: int = 0
    num_samples: int = 10
    max_length: int = 20


def stoi(s: str) -> int:
    return 0 if s == "." else ord(s) - ord("a") + 1


def itos(i: int) -> str:
    return "." if i == 0 else chr(i + ord("a") - 1)


def sample(
    probs: jnp.ndarray,
    key: jax.Array,
    num_samples: int = 10,
    max_length: int = 20,
) -> list[str]:
    """Generates names character-by-character autoregressively until '.' or max_length."""
    generated_names = []
    for _ in range(num_samples):
        cur_name = ""
        idx = 0
        for _ in range(max_length):
            key, subkey = jax.random.split(key)
            next_probs = probs[idx].astype(jnp.float32)
            idx = int(jnp.argmax(jax.random.multinomial(subkey, n=1, p=next_probs)))
            if idx == 0:
                break
            cur_name += itos(idx)
        generated_names.append(cur_name)
    return generated_names


def main(config: BigramConfig = BigramConfig()):
    print("=" * 60)
    print("  jakemore: Bigram Count Language Model (JAX)")
    print("=" * 60)

    # 1. Load dataset
    with open(config.data_path, "r", encoding="utf-8") as f:
        names = [line.strip().lower() for line in f if line.strip()]

    # 2. Count bigram transitions
    bigram_counts = Counter()
    for name in names:
        seq = ["."] + list(name) + ["."]
        bigrams = [(i, j) for i, j in zip(seq[:-1], seq[1:])]
        bigram_counts.update(bigrams)

    total_transitions = sum(bigram_counts.values())
    print(f"Loaded {len(names):,} names ({total_transitions:,} bigram transitions)")

    # 3. Build count matrix and apply Laplace smoothing
    bigram_model_counts = jnp.zeros((27, 27), dtype=jnp.float32)
    for (s1, s2), count in bigram_counts.items():
        i1, i2 = stoi(s1), stoi(s2)
        bigram_model_counts = bigram_model_counts.at[i1, i2].set(float(count))

    smoothed_counts = bigram_model_counts + config.smoothing
    bigram_model_probs = smoothed_counts / smoothed_counts.sum(axis=1, keepdims=True)
    bigram_model_logprobs = jnp.log(bigram_model_probs)

    # 4. Evaluate dataset Negative Log-Likelihood (NLL) and Perplexity
    nll = 0.0
    for (s1, s2), count in bigram_counts.items():
        i1, i2 = stoi(s1), stoi(s2)
        nll -= float(bigram_model_logprobs[i1, i2]) * count
    avg_nll = nll / total_transitions
    perplexity = float(jnp.exp(avg_nll))

    print(f"Laplace Smoothing:  {config.smoothing}")
    print(f"Average Train NLL:  {avg_nll:.4f}")
    print(f"Dataset Perplexity: {perplexity:.4f}")

    # 5. Generate sample names
    master_key = jax.random.key(config.seed)
    generated = sample(
        bigram_model_probs,
        master_key,
        num_samples=config.num_samples,
        max_length=config.max_length,
    )

    print("\nGenerated Names:")
    for idx, name in enumerate(generated, start=1):
        print(f"  {idx:2d}. {name}")

    # 6. Save model
    os.makedirs(config.save_dir, exist_ok=True)
    save_path = os.path.join(config.save_dir, "bigram_model.pkl")
    model_data = {
        "counts": bigram_model_counts,
        "probs": bigram_model_probs,
        "logprobs": bigram_model_logprobs,
        "smoothing": config.smoothing,
        "nll": avg_nll,
        "perplexity": perplexity,
    }
    with open(save_path, "wb") as f:
        pickle.dump(model_data, f)
    print(f"\nModel saved to '{save_path}'\n")


if __name__ == "__main__":
    main()
