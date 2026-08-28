import os
import grain.python as grain
import jax.numpy as jnp
import numpy as np

VOCAB_SIZE: int = 27
PAD_CHAR: str = "."


def stoi(s: str) -> int:
    """Encodes character to integer token ('.' -> 0, 'a'-'z' -> 1-26)."""
    return 0 if s == PAD_CHAR else ord(s) - ord("a") + 1


def itos(i: int) -> str:
    """Decodes integer token to character (0 -> '.', 1-26 -> 'a'-'z')."""
    return PAD_CHAR if i == 0 else chr(i + ord("a") - 1)


def load_names(data_path: str) -> list[str]:
    """Loads and cleans dataset of names."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at '{data_path}'")
    with open(data_path, "r", encoding="utf-8") as f:
        names = [line.strip().lower() for line in f if line.strip()]
    return names


def build_tokenized_dataset(
    names: list[str], ngram_size: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Builds (X, Y) context-target token pairs per name with '.' padding."""
    if ngram_size < 2:
        raise ValueError(f"ngram_size must be at least 2, got {ngram_size}")

    xs: list[list[int]] = []
    ys: list[int] = []
    pad = [0] * (ngram_size - 1)

    for name in names:
        tokens = pad + [stoi(c) for c in name] + [0]
        for i in range(len(tokens) - ngram_size + 1):
            xs.append(tokens[i : i + ngram_size - 1])
            ys.append(tokens[i + ngram_size - 1])

    return jnp.array(xs, dtype=jnp.int32), jnp.array(ys, dtype=jnp.int32)


class NgramDatasetSource:
    """Grain MapDataset source for (X, Y) pairs."""

    def __init__(self, xs: jnp.ndarray | np.ndarray, ys: jnp.ndarray | np.ndarray):
        self.xs = np.array(xs)
        self.ys = np.array(ys)

    def __len__(self) -> int:
        return len(self.xs)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        return {"x": self.xs[idx], "y": self.ys[idx]}


def create_grain_loader(
    xs: jnp.ndarray | np.ndarray,
    ys: jnp.ndarray | np.ndarray,
    batch_size: int,
    seed: int = 42,
    repeat: bool = True,
    shuffle: bool = True,
):
    """Creates a Grain DataLoader iterator yielding mini-batches."""
    source = NgramDatasetSource(xs, ys)
    ds = grain.MapDataset.source(source)
    if shuffle:
        ds = ds.shuffle(seed=seed)
    if repeat:
        ds = ds.repeat()
    ds = ds.batch(batch_size=batch_size, drop_remainder=False)
    return ds.to_iter_dataset()
