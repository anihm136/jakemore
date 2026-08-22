import os
import pickle
from dataclasses import dataclass
from typing import Optional
import jax
import jax.numpy as jnp


@dataclass
class NNConfig:
    """Configuration for the single-layer neural network bigram model."""

    data_path: str = "names.txt"
    save_dir: str = "models"
    learning_rate: float = 10.0
    num_steps: int = 100
    batch_size: Optional[int] = None  # None for full-batch GD; int for mini-batch SGD
    reg_weight: float = 0.01  # L2 regularization weight
    seed: int = 0
    num_samples: int = 10
    temperature: float = 1.0
    max_length: int = 20
    log_interval: int = 10


def stoi(s: str) -> int:
    return 0 if s == "." else ord(s) - ord("a") + 1


def itos(i: int) -> str:
    return "." if i == 0 else chr(i + ord("a") - 1)


def build_encoded_dataset(
    char_seq: list[str], num_classes: int = 27, batch_size: Optional[int] = None
):
    enc_seq = [stoi(i) for i in char_seq]
    feat, label = enc_seq[:-1], enc_seq[1:]
    feat_arr, label_arr = jnp.array(feat), jnp.array(label)

    xs = jax.nn.one_hot(feat_arr, num_classes=num_classes, dtype=jnp.float32)
    ys = jax.nn.one_hot(label_arr, num_classes=num_classes, dtype=jnp.float32)

    if batch_size is None:
        batch_size = xs.shape[0]

    arrlen = batch_size * (xs.shape[0] // batch_size)
    xs = xs[:arrlen].reshape((-1, batch_size, num_classes))
    ys = ys[:arrlen].reshape((-1, batch_size, num_classes))

    return xs, ys


def forward(x: jnp.ndarray, W: jnp.ndarray) -> jnp.ndarray:
    logits = x @ W
    return logits


def logits_to_logprobs(logits: jnp.ndarray) -> jnp.ndarray:
    counts = jnp.exp(logits)
    probs = counts / counts.sum(axis=1, keepdims=True)
    logprobs = probs.log()
    return logprobs


def nll(logprobs: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    loss = -jnp.sum(logprobs * labels) / len(labels)
    return loss


@jax.jit
def train_step(
    W: jnp.ndarray,
    xs: jnp.ndarray,
    ys: jnp.ndarray,
    lr: float = 10.0,
    reg_weight: float = 0.01,
):
    def loss_func(W, x, y):
        logits = forward(x, W)
        logprobs = logits_to_logprobs(logits)
        loss = nll(logprobs, y) + reg_weight * (W**2).mean()
        return loss

    loss, grad = jax.value_and_grad(loss_func)(W, xs, ys)
    W -= lr * grad
    return W, loss


def sample_nn(
    W: jnp.ndarray,
    key: jax.Array,
    num_samples: int = 10,
    max_length: int = 20,
    temperature: float = 1.0,
) -> list[str]:
    """Generates names from the neural network weights autoregressively."""
    probs = jax.nn.softmax(W / temperature, axis=-1)
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


def main(config: NNConfig = NNConfig()):
    print("=" * 60)
    print("  jakemore: Bigram Neural Network (JAX)")
    print("=" * 60)

    # 1. Load dataset & build continuous token stream
    with open(config.data_path, "r", encoding="utf-8") as f:
        names = [line.strip().lower() for line in f if line.strip()]

    char_seq = ["."] + [c for name in names for c in (list(name) + ["."])]
    xs, ys = build_encoded_dataset(char_seq, batch_size=config.batch_size)

    total_examples = xs.shape[0] * xs.shape[1]
    print(f"Loaded {len(names):,} names ({total_examples:,} bigram transitions)")
    print(
        f"Training Mode: {'Full-batch GD' if config.batch_size is None else f'Mini-batch SGD (B={config.batch_size})'}"
    )
    print(
        f"Learning Rate: {config.learning_rate} | Regularization (λ): {config.reg_weight} | Steps: {config.num_steps}\n"
    )

    # 2. Initialize weights
    master_key = jax.random.key(config.seed)
    master_key, key = jax.random.split(master_key)
    W = jax.random.normal(key, (27, 27))

    # 3. Train network
    print("Starting training:")
    for step in range(1, config.num_steps + 1):
        for x_batch, y_batch in zip(xs, ys):
            W, loss = train_step(
                W,
                x_batch,
                y_batch,
                lr=config.learning_rate,
                reg_weight=config.reg_weight,
            )
        if step % config.log_interval == 0 or step == 1 or step == config.num_steps:
            print(f"  Step {step:3d}/{config.num_steps} | Loss: {loss.item():.4f}")

    # 4. Final Evaluation (NLL without regularization penalty)
    all_logits = forward(xs.reshape(-1, 27), W)
    all_logprobs = logits_to_logprobs(all_logits)
    final_nll = float(nll(all_logprobs, ys.reshape(-1, 27)))
    perplexity = float(jnp.exp(final_nll))

    print(f"\nFinal Train NLL:    {final_nll:.4f}")
    print(f"Dataset Perplexity: {perplexity:.4f}")

    # 5. Generate sample names
    master_key, sample_key = jax.random.split(master_key)
    generated = sample_nn(
        W,
        sample_key,
        num_samples=config.num_samples,
        max_length=config.max_length,
        temperature=config.temperature,
    )

    print("\nGenerated Names:")
    for idx, name in enumerate(generated, start=1):
        print(f"  {idx:2d}. {name}")

    # 6. Save model
    os.makedirs(config.save_dir, exist_ok=True)
    save_path = os.path.join(config.save_dir, "nn_model.pkl")
    model_data = {
        "W": W,
        "lr": config.learning_rate,
        "reg_weight": config.reg_weight,
        "nll": final_nll,
        "perplexity": perplexity,
    }
    with open(save_path, "wb") as f:
        pickle.dump(model_data, f)
    print(f"\nModel saved to '{save_path}'\n")


if __name__ == "__main__":
    main()
