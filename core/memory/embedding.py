import logging
from typing import Any, List, Optional

from core.config import MemoryVectorConfig
from core.llm.client import create_openai_client
from core.utils import log_exception

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
        self.client: Optional[Any] = None

        if not self._ready:
            return

        try:
            self.client = create_openai_client(api_key=cfg.embedding_api_key, base_url=cfg.embedding_api_url)
        except Exception:
            log_exception(
                logger,
                "memory.embedding.init.error",
                "Embedding 初始化失败，回退到关键词检索",
                component="memory",
                fallback="keyword_only",
            )
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
        except Exception:
            log_exception(
                logger,
                "memory.embedding.request.error",
                "Embedding 请求失败",
                component="memory",
                model=self.cfg.embedding_model,
                text_len=len(payload),
            )
            return None
