"""Embeddings wrapper that reports progress while vectors are created."""

import time

from langchain_core.embeddings import Embeddings


class ProgressEmbeddings(Embeddings):
    """Wraps an embeddings model so every vector is printed as it is created.

    Bedrock's Titan API embeds one text per request, so a large batch means many
    sequential calls with no output. This makes that progress visible.
    """

    def __init__(self, inner, preview_values=4):
        self.inner = inner
        self.preview_values = preview_values

    def embed_documents(self, texts):
        total = len(texts)
        started = time.perf_counter()
        vectors = []

        for i, text in enumerate(texts, start=1):
            vector = self.inner.embed_query(text)
            vectors.append(vector)

            head = ", ".join(f"{v:+.4f}" for v in vector[:self.preview_values])
            snippet = " ".join(text.split())[:55]
            elapsed = time.perf_counter() - started
            eta = (elapsed / i) * (total - i)
            print(
                f"[{i:>4}/{total}] dim={len(vector)} "
                f"[{head}, ...] eta {eta:5.1f}s | {snippet}..."
            )

        print(f"Embedded {total} chunks in {time.perf_counter() - started:.1f}s")
        return vectors

    def embed_query(self, text):
        return self.inner.embed_query(text)
