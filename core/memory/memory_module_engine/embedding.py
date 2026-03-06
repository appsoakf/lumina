"""嵌入生成器抽象层"""
from collections import OrderedDict
from threading import RLock
from abc import ABC, abstractmethod
from typing import List


class EmbeddingProvider(ABC):
    """嵌入生成器接口"""

    @abstractmethod
    def encode(self, text: str) -> List[float]:
        """生成文本嵌入"""
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """返回向量维度"""
        pass


class OpenAIEmbedding(EmbeddingProvider):
    """OpenAI API（支持兼容接口）"""

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-large",
        dimensions: int = 3072,
        base_url: str = None,
        cache_enabled: bool = True,
        cache_max_entries: int = 4096,
    ):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.dimensions = dimensions
        self.cache_enabled = bool(cache_enabled)
        self.cache_max_entries = max(int(cache_max_entries), 1)
        self._cache: OrderedDict[str, List[float]] = OrderedDict()
        self._cache_lock = RLock()

    def _cache_key(self, text: str) -> str:
        # 缓存 key 使用模型与维度，避免多模型场景冲突。
        return f"{self.model}|{self.dimensions}|{text}"

    def encode(self, text: str) -> List[float]:
        if self.cache_enabled:
            key = self._cache_key(text)
            with self._cache_lock:
                cached = self._cache.get(key)
                if cached is not None:
                    self._cache.move_to_end(key)
                    # 返回副本，避免外部修改污染缓存。
                    return list(cached)

        response = self.client.embeddings.create(
            model=self.model,
            input=text,
            dimensions=self.dimensions
        )
        embedding = list(response.data[0].embedding)

        if self.cache_enabled:
            with self._cache_lock:
                self._cache[key] = embedding
                self._cache.move_to_end(key)
                while len(self._cache) > self.cache_max_entries:
                    self._cache.popitem(last=False)

        return embedding

    def get_dimension(self) -> int:
        return self.dimensions
