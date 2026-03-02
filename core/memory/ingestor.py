import re
from typing import Dict, List


class MemoryIngestor:
    """Extracts structured memory candidates from user utterances."""

    PROFILE_PATTERNS = [
        re.compile(r"我喜欢(.+)$"),
        re.compile(r"我不喜欢(.+)$"),
        re.compile(r"我的偏好是(.+)$"),
        re.compile(r"记住[:：]?(.+)$"),
    ]

    COMMITMENT_PATTERNS = [
        re.compile(r"(?:提醒我|记得|待办[:：]?)(.+)$"),
        re.compile(r"(.+?)(?:截止|在)(\d{1,2}月\d{1,2}日|\d{4}[/-]\d{1,2}[/-]\d{1,2})"),
    ]

    def extract_profile_candidates(self, text: str) -> List[Dict]:
        candidates = []
        source = text.strip()
        for p in self.PROFILE_PATTERNS:
            m = p.search(source)
            if m:
                value = m.group(1).strip()
                if value:
                    candidates.append({"content": value, "tags": "profile,preference"})
        return candidates

    def extract_commitment_candidates(self, text: str) -> List[Dict]:
        candidates = []
        source = text.strip()

        for p in self.COMMITMENT_PATTERNS:
            m = p.search(source)
            if not m:
                continue

            if len(m.groups()) == 1:
                todo = m.group(1).strip()
                due = ""
            else:
                todo = m.group(1).strip()
                due = m.group(2).strip()

            if todo:
                payload = {"status": "open", "due": due}
                candidates.append({"content": todo, "tags": "commitment,todo", "payload": payload})

        return candidates
