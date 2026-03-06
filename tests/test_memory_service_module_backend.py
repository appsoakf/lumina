import os
import shutil
import sys
import unittest
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.memory.service import MemoryService
from core.config import load_app_config


class MemoryServiceModuleBackendTests(unittest.TestCase):
    def setUp(self):
        self._prev_runtime = os.environ.get("LUMINA_RUNTIME_DIR")
        self._prev_vector_enabled = os.environ.get("LUMINA_MEMORY_VECTOR_ENABLED")

        self.temp_runtime = PROJECT_ROOT / "runtime" / f"lumina-test-memory-{uuid.uuid4().hex[:8]}"
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

    def test_record_session_round_and_recent_history(self):
        self.memory.record_session_round(
            session_id="s1",
            user_text="你好",
            assistant_reply="你好，我在。",
            metadata={"round": 1},
        )

        history = self.memory.get_recent_history("s1")
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["role"], "user")
        self.assertEqual(history[1]["role"], "assistant")

    def test_ingest_turn_persists_memory_and_build_context(self):
        self.memory.ingest_turn(
            session_id="s2",
            user_text="我喜欢博物馆和清淡饮食，提醒我下周提交周报。",
            assistant_reply="好的，我会记住这些偏好并提醒你。",
            meta={},
        )

        context = self.memory.build_context(query="博物馆 周报")
        stats = self.memory._engine.get_stats()

        self.assertGreater(stats.get("working_count", 0) + stats.get("long_term_count", 0), 0)
        self.assertTrue(context)
        self.assertTrue(any(k in context for k in ["博物馆", "清淡饮食", "周报"]))

    def test_ingest_turn_stores_procedural_task_template(self):
        self.memory.ingest_turn(
            session_id="s3",
            user_text="帮我做任务复盘",
            assistant_reply="已完成复盘",
            meta={
                "task_mode": True,
                "task_id": "task-001",
                "task_error": False,
                "plan": {"goal": "北京三日游规划"},
            },
        )

        results = self.memory._engine.search("北京三日游规划", top_k=6)
        self.assertTrue(any("procedural:" in item.content for item in results))


if __name__ == "__main__":
    unittest.main()
