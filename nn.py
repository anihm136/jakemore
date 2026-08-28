import os
import time
import pickle
from dataclasses import dataclass
from typing import Optional
import jax
import jax.numpy as jnp

from common import (
    VOCAB_SIZE,
    build_tokenized_dataset,
    create_grain_loader,
    itos,
    load_names,
)


@dataclass
class NNConfig:
    """Configuration for the MLP character language model."""

    data_path: str = "names.txt"
    save_dir: str = "models"
    ngram_size: int = 4
    emb_size: Optional[int] = 10  # None for one-hot; int for embedding table
    hidden_dim: int = 200
    learning_rate: float = 0.1
    lr_decay_step: float = 0.8  # Fraction of steps after which LR is decayed
    lr_decay_factor: float = 0.1
    num_steps: int = 10000
    batch_size: int = 32
    reg_weight: float = 0.01  # L2 regularization weight
    train_ratio: float = 0.8
    seed: int = 0
    num_samples: int = 5
    max_length: int = 20
    log_interval: int = 200


def init_mlp_params(
    key: jax.Array,
    context_len: int,
    vocab_size: int,
    hidden_dim: int,
    emb_size: Optional[int],
) -> dict[str, jnp.ndarray]:
    key_emb, key_w1, key_w2 = jax.random.split(key, 3)

    model: dict[str, jnp.ndarray] = {}
    if emb_size is not None:
        model["emb"] = jax.random.normal(key_emb, (vocab_size, emb_size))
        in_dim = context_len * emb_size
    else:
        in_dim = context_len * vocab_size

    model["W1"] = jax.random.normal(key_w1, (in_dim, hidden_dim))
    model["b1"] = jnp.zeros((hidden_dim,), dtype=jnp.float32)

    model["W2"] = jax.random.normal(key_w2, (hidden_dim, vocab_size))
    model["b2"] = jnp.zeros((vocab_size,), dtype=jnp.float32)

    return model


def forward(x: jnp.ndarray, model: dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Computes unnormalized logits for a single context sequence of token IDs."""
    if "emb" in model:
        h_in = model["emb"][x]
    else:
        h_in = jax.nn.one_hot(
            x, num_classes=model["W1"].shape[0] // len(x), dtype=jnp.float32
        )

    h_in = h_in.reshape(-1)
    h = jnp.tanh(h_in @ model["W1"] + model["b1"])
    logits = h @ model["W2"] + model["b2"]
    return logits


def forward_batch(x_batch: jnp.ndarray, model: dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Vectorized forward pass across a batch using jax.vmap."""
    return jax.vmap(forward, in_axes=(0, None))(x_batch, model)


def loss_fn(
    model: dict[str, jnp.ndarray],
    x_batch: jnp.ndarray,
    y_batch: jnp.ndarray,
    reg_weight: float,
) -> jnp.ndarray:
    """Cross-entropy loss with L2 regularization."""
    logits = forward_batch(x_batch, model)
    logprobs = jax.nn.log_softmax(logits, axis=-1)
    target_logprobs = jnp.take_along_axis(logprobs, y_batch[:, None], axis=1)
    nll_loss = -jnp.mean(target_logprobs)

    reg_loss = (
        0.5
        * (jnp.sum(model["W1"] ** 2) + jnp.sum(model["W2"] ** 2))
        / (model["W1"].size + model["W2"].size)
    )
    return nll_loss + reg_weight * reg_loss


@jax.jit
def train_step(
    model: dict[str, jnp.ndarray],
    x_batch: jnp.ndarray,
    y_batch: jnp.ndarray,
    lr: float,
    reg_weight: float,
) -> tuple[dict[str, jnp.ndarray], jnp.ndarray]:
    """Single JIT-compiled optimization step."""
    loss, grads = jax.value_and_grad(loss_fn)(model, x_batch, y_batch, reg_weight)
    new_model = jax.tree.map(lambda p, g: p - lr * g, model, grads)
    return new_model, loss


@jax.jit
def eval_batch_nll(
    model: dict[str, jnp.ndarray], x_batch: jnp.ndarray, y_batch: jnp.ndarray
) -> jnp.ndarray:
    """Computes total NLL sum for a batch."""
    logits = forward_batch(x_batch, model)
    logprobs = jax.nn.log_softmax(logits, axis=-1)
    target_logprobs = jnp.take_along_axis(logprobs, y_batch[:, None], axis=1)
    return -jnp.sum(target_logprobs)


def evaluate(
    model: dict[str, jnp.ndarray],
    xs: jnp.ndarray,
    ys: jnp.ndarray,
    batch_size: int = 1024,
) -> tuple[float, float]:
    """Evaluates NLL and perplexity in batches over a dataset split."""
    total_nll = 0.0
    n_samples = xs.shape[0]
    for i in range(0, n_samples, batch_size):
        xb = xs[i : i + batch_size]
        yb = ys[i : i + batch_size]
        total_nll += float(eval_batch_nll(model, xb, yb))
    mean_nll = total_nll / n_samples
    perplexity = float(jnp.exp(mean_nll))
    return mean_nll, perplexity


def sample_nn(
    model: dict[str, jnp.ndarray],
    key: jax.Array,
    context_len: int,
    num_samples: int = 10,
    max_length: int = 20,
) -> list[str]:
    """Generates names autoregressively from the neural network."""
    generated_names = []
    for _ in range(num_samples):
        cur_tokens = [0] * context_len
        out_chars = []
        for _ in range(max_length):
            key, subkey = jax.random.split(key)
            context = jnp.array(cur_tokens[-context_len:], dtype=jnp.int32)
            logits = forward(context, model)
            next_token = int(jax.random.categorical(subkey, logits))
            if next_token == 0:
                break
            out_chars.append(itos(next_token))
            cur_tokens.append(next_token)
        generated_names.append("".join(out_chars))
    return generated_names


def main(config: NNConfig = NNConfig()):
    print("=" * 60)
    print(f"  jakemore: {config.ngram_size}-gram MLP Language Model (JAX)")
    print("=" * 60)

    # 1. Load dataset & build tokenized n-grams
    names = load_names(config.data_path)
    xs, ys = build_tokenized_dataset(names, ngram_size=config.ngram_size)

    # Train / test split
    split_idx = int(round(xs.shape[0] * config.train_ratio))
    train_xs, train_ys = xs[:split_idx], ys[:split_idx]
    test_xs, test_ys = xs[split_idx:], ys[split_idx:]

    print(
        f"Loaded {len(names):,} names ({xs.shape[0]:,} {config.ngram_size}-gram transitions)"
    )
    print(f"Train set: {train_xs.shape[0]:,} transitions")
    print(f"Test set:  {test_xs.shape[0]:,} transitions")
    print(
        f"Embedding: {f'{config.emb_size}-dim table' if config.emb_size is not None else 'One-hot'}"
    )
    print(f"Hidden Dim: {config.hidden_dim} (tanh)")
    print(
        f"LR: {config.learning_rate} (decay {config.lr_decay_factor}x @ {int(config.lr_decay_step * 100)}%) | Steps: {config.num_steps:,} | Batch: {config.batch_size}\n"
    )

    # 2. Initialize parameters
    init_key = jax.random.key(config.seed)
    init_key, master_key = jax.random.split(init_key)
    model = init_mlp_params(
        init_key,
        context_len=config.ngram_size - 1,
        emb_size=config.emb_size,
        hidden_dim=config.hidden_dim,
        vocab_size=VOCAB_SIZE,
    )
    param_count = sum(p.size for p in jax.tree.flatten(model)[0])
    print(f"Model Parameters: {param_count:,}\n")

    # 3. Train network using Grain DataLoader
    train_loader = create_grain_loader(
        train_xs,
        train_ys,
        batch_size=config.batch_size,
        seed=config.seed,
        repeat=True,
    )
    train_iter = iter(train_loader)

    print("Training:")
    start = time.perf_counter()
    for step in range(1, config.num_steps + 1):
        batch = next(train_iter)
        xb = jnp.array(batch["x"], dtype=jnp.int32)
        yb = jnp.array(batch["y"], dtype=jnp.int32)

        lr = (
            config.learning_rate
            if (step / config.num_steps) < config.lr_decay_step
            else config.learning_rate * config.lr_decay_factor
        )

        model, loss = train_step(
            model,
            xb,
            yb,
            lr=lr,
            reg_weight=config.reg_weight,
        )

        if step % config.log_interval == 0 or step == config.num_steps:
            print(
                f"  Step {step:6d}/{config.num_steps:,} | Train Loss: {loss.item():.4f}"
            )

    stop = time.perf_counter()
    train_time = stop - start

    # 4. Evaluate NLL & Perplexity
    train_nll, train_perplexity = evaluate(model, train_xs, train_ys)
    test_nll, test_perplexity = evaluate(model, test_xs, test_ys)

    print(f"\nTrain Time:         {train_time:.2f} s")
    print(
        f"Train NLL:          {train_nll:.4f} | Train Perplexity: {train_perplexity:.4f}"
    )
    print(
        f"Test NLL:           {test_nll:.4f} | Test Perplexity:  {test_perplexity:.4f}"
    )

    # 5. Generate sample names
    master_key, sample_key = jax.random.split(master_key)
    generated = sample_nn(
        model,
        sample_key,
        context_len=config.ngram_size - 1,
        num_samples=config.num_samples,
        max_length=config.max_length,
    )

    print("\nGenerated Names:")
    for idx, name in enumerate(generated, start=1):
        print(f"  {idx:2d}. {name}")

    # 6. Save model
    os.makedirs(config.save_dir, exist_ok=True)
    save_path = os.path.join(config.save_dir, "nn_model.pkl")
    model_data = {
        "model": model,
        "config": config,
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
