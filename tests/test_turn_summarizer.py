import os
import shutil
import sys
import unittest
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import load_app_config
from core.memory.service import MemoryService


class MemoryIngestExtractionTests(unittest.TestCase):
    def setUp(self):
        self._prev_runtime = os.environ.get("LUMINA_RUNTIME_DIR")
        self._prev_vector_enabled = os.environ.get("LUMINA_MEMORY_VECTOR_ENABLED")

        self.temp_runtime = PROJECT_ROOT / "runtime" / f"lumina-test-ingest-{uuid.uuid4().hex[:8]}"
        os.environ["LUMINA_RUNTIME_DIR"] = str(self.temp_runtime)
        os.environ["LUMINA_MEMORY_VECTOR_ENABLED"] = "0"
        load_app_config.cache_clear()

        self.memory = MemoryService(short_history_limit=16)

    def tearDown(self):
        try:
            self.memory.close()
        except Exception:
            pass

        if self._prev_runtime is None:
            os.environ.pop("LUMINA_RUNTIME_DIR", None)
        else:
            os.environ["LUMINA_RUNTIME_DIR"] = self._prev_runtime

        if self._prev_vector_enabled is None:
            os.environ.pop("LUMINA_MEMORY_VECTOR_ENABLED", None)
        else:
            os.environ["LUMINA_MEMORY_VECTOR_ENABLED"] = self._prev_vector_enabled

        load_app_config.cache_clear()
        shutil.rmtree(self.temp_runtime, ignore_errors=True)

    def test_ingest_extracts_profile_and_topic_context(self):
        self.memory.ingest_turn(
            session_id="s1",
            user_text="帮我规划北京三日游。另外我喜欢博物馆和清淡饮食。",
            assistant_reply="好的，我会先整理行程。",
            meta={},
        )

        context = self.memory.build_context(query="北京三日游 博物馆")
        self.assertTrue(context)
        self.assertIn("用户偏好", context)
        self.assertTrue(any(k in context for k in ["博物馆", "清淡饮食", "北京三日游"]))

    def test_ingest_extracts_commitment_context(self):
        self.memory.ingest_turn(
            session_id="s2",
            user_text="提醒我明天提交周报，截止2026-03-07。",
            assistant_reply="收到，我会提醒你。",
            meta={},
        )

        context = self.memory.build_context(query="提交周报")
        self.assertTrue(context)
        self.assertIn("未完成事项", context)
        self.assertIn("提交周报", context)


if __name__ == "__main__":
    unittest.main()
