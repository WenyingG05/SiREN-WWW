"""Shared batch embedding encoder and LRU cache, independent of belief storage."""
from collections import OrderedDict
import numpy as np

class EmbeddingModel:
    """
    Encode text with mean pooling and cache recent embeddings.
    Repeated text is encoded once until its entry leaves the LRU cache.
    """

    def __init__(self, model_name: str, cache_size: int = 4096):
        import torch
        from transformers import AutoTokenizer, AutoModel
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_size = cache_size

    def _mean_pool(self, model_output, attention_mask) -> np.ndarray:
        import torch
        token_emb = model_output.last_hidden_state          # (B, L, D)
        mask = attention_mask.unsqueeze(-1).float()         # (B, L, 1)
        summed = (token_emb * mask).sum(dim=1)              # (B, D)
        counts = mask.sum(dim=1).clamp(min=1e-9)            # (B, 1)
        return (summed / counts).cpu().numpy()              # (B, D)

    def _normalize(self, vecs: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-9)
        return vecs / norms

    def _lru_put(self, text: str, vec: np.ndarray):
        if text in self._cache:
            self._cache.move_to_end(text)
        else:
            if len(self._cache) >= self._cache_size:
                self._cache.popitem(last=False)
            self._cache[text] = vec

    def encode(self, text: str) -> np.ndarray:
        """Return a normalized embedding, using the cache when available."""
        if text in self._cache:
            self._cache.move_to_end(text)
            return self._cache[text]
        vec = self.encode_batch([text])[0]
        self._lru_put(text, vec)
        return vec

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """
        Return normalized embeddings with shape (n, dim).
        Encode cache misses together, then update the cache.
        """
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        result = np.empty((len(texts),), dtype=object)
        need_encode: list[int] = []   # Indices of cache misses.

        for i, t in enumerate(texts):
            if t in self._cache:
                self._cache.move_to_end(t)
                result[i] = self._cache[t]
            else:
                need_encode.append(i)

        if need_encode:
            batch_texts = [texts[i] for i in need_encode]
            enc = self.tokenizer(
                batch_texts, padding=True, truncation=True,
                max_length=128, return_tensors="pt"
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            import torch
            with torch.no_grad():
                out = self.model(**enc)
            vecs = self._mean_pool(out, enc["attention_mask"])
            vecs = self._normalize(vecs)          # (n_new, D)
            for j, idx in enumerate(need_encode):
                v = vecs[j]
                result[idx] = v
                self._lru_put(texts[idx], v)

        return np.stack(result.tolist(), axis=0).astype(np.float32)

    @staticmethod
    def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Compute cosine similarity as the dot product of normalized vectors."""
        return float(np.dot(vec_a, vec_b))


