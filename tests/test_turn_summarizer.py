import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.memory.turn_summarizer import AsyncTurnSummarizer, TurnSummaryExtractor


class TurnSummarizerTests(unittest.TestCase):
    def test_extractor_extracts_topic_and_profiles(self):
        extractor = TurnSummaryExtractor()
        summary = extractor.summarize(
            user_text="帮我规划北京三日游。另外我喜欢博物馆和清淡饮食。",
            assistant_reply="好的，我会先整理行程。",
        )

        self.assertTrue(summary.topic)
        self.assertIn("规划北京三日游", summary.topic)
        self.assertTrue(any("博物馆" in item for item in summary.profile_candidates))
        self.assertTrue(any("清淡饮食" in item for item in summary.profile_candidates))

    def test_async_summarizer_processes_queue(self):
        received = []

        def on_summary(summary, item):
            received.append((summary, item))

        summarizer = AsyncTurnSummarizer(
            extractor=TurnSummaryExtractor(),
            on_summary=on_summary,
            enabled=True,
            queue_size=8,
        )
        try:
            ok = summarizer.enqueue(
                {
                    "session_id": "s1",
                    "user_id": "u1",
                    "user_text": "我喜欢慢跑，今天主要聊健身计划。",
                    "assistant_reply": "明白了，我们可以按周安排。",
                    "meta": {},
                }
            )
            self.assertTrue(ok)
        finally:
            summarizer.close()

        self.assertEqual(len(received), 1)
        summary, item = received[0]
        self.assertEqual(item.get("session_id"), "s1")
        self.assertTrue(summary.topic)
        self.assertTrue(any("慢跑" in x for x in summary.profile_candidates))


if __name__ == "__main__":
    unittest.main()
