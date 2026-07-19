"""Sentence-transformer wrapper. Vectors come out L2-normalized so downstream
similarity is a plain dot product."""

import numpy as np
from sentence_transformers import SentenceTransformer


def load_model(model_name, device):
    return SentenceTransformer(model_name, device=device)


def embed_texts(model, texts, batch_size):
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vectors.astype(np.float32)
