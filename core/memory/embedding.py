import logging
from typing import List, Optional

from openai import OpenAI

from core.config import MemoryVectorConfig

logger = logging.getLogger(__name__)


class EmbeddingProvider:
    def is_ready(self) -> bool:
        return False

    def embed(self, text: str) -> Optional[List[float]]:
        raise NotImplementedError


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, cfg: MemoryVectorConfig):
        self.cfg = cfg
        self._ready = bool(cfg.enabled)
        self.client: Optional[OpenAI] = None

        if not self._ready:
            return

        try:
            self.client = OpenAI(api_key=cfg.embedding_api_key, base_url=cfg.embedding_api_url)
        except Exception as exc:
            logger.warning(f"Embedding provider init failed, fallback to keyword-only retrieval: {exc}")
            self._ready = False

    def is_ready(self) -> bool:
        return self._ready and self.client is not None

    def embed(self, text: str) -> Optional[List[float]]:
        if not self.is_ready():
            return None

        payload = (text or "").strip()
        if not payload:
            return None

        try:
            response = self.client.embeddings.create(model=self.cfg.embedding_model, input=payload)
            if not response.data:
                return None
            vector = response.data[0].embedding
            if not isinstance(vector, list) or not vector:
                return None
            return [float(x) for x in vector]
        except Exception as exc:
            logger.warning(f"Embedding request failed: {exc}")
            return None
