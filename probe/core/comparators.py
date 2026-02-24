from abc import ABC, abstractmethod
import numpy as np
import logging
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


class Comparator(ABC):
    @abstractmethod
    def similarity(self, text_a: str, text_b: str) -> float:
        ...


class EmbeddingSimilarity(Comparator):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)

    def similarity(self, text_a: str, text_b: str) -> float:
        self._load()
        embs = self._model.encode([text_a, text_b], normalize_embeddings=True)
        return max(0.0, min(1.0, float(np.dot(embs[0], embs[1]))))

    def batch_similarity(self, reference: str, candidates: list[str]) -> list[float]:
        self._load()
        all_texts = [reference] + candidates
        embs = self._model.encode(all_texts, normalize_embeddings=True)
        ref = embs[0]
        return [max(0.0, min(1.0, float(np.dot(ref, embs[i+1])))) for i in range(len(candidates))]


class ExactMatch(Comparator):
    def similarity(self, text_a: str, text_b: str) -> float:
        return 1.0 if text_a.strip().lower() == text_b.strip().lower() else 0.0


class ContainsMatch(Comparator):
    def __init__(self, extract_fn=None):
        self._extract = extract_fn or (lambda x: x.strip().lower())

    def similarity(self, text_a: str, text_b: str) -> float:
        a, b = self._extract(text_a), self._extract(text_b)
        if not a or not b:
            return 0.0
        return 1.0 if (a in b or b in a) else 0.0
