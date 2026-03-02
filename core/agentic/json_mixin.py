import json
import re
from typing import Any, Dict


class JSONParseMixin:
    """Shared helpers for model outputs that should contain JSON."""

    def clean_json_text(self, text: str) -> str:
        cleaned = (text or "").strip()
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
        return cleaned

    def parse_json_object(self, text: str, *, allow_brace_extract: bool = True) -> Dict[str, Any]:
        cleaned = self.clean_json_text(text)
        try:
            payload = json.loads(cleaned)
            return payload if isinstance(payload, dict) else {}
        except json.JSONDecodeError:
            if not allow_brace_extract:
                raise
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1 and end > start:
                payload = json.loads(cleaned[start : end + 1])
                return payload if isinstance(payload, dict) else {}
            raise
