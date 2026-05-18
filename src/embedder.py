"""Embedding and similarity utils powered by sentence-transformers."""

import numpy as np
from sentence_transformers import SentenceTransformer
from .config import EMBEDDING_MODEL

_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def encode_texts(texts: list[str], batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
    model = get_model()
    return model.encode(texts, batch_size=batch_size, show_progress_bar=show_progress, convert_to_numpy=True)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity — a: (d,) or (n,d), b: (m,d). Returns (n,m) or (m,)."""
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return a_norm @ b_norm.T
