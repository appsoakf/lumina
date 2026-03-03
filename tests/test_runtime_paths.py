import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import ROOT_DIR
from core.paths import backups_root, runtime_root


class RuntimePathTests(unittest.TestCase):
    def test_runtime_root_defaults_to_project_runtime(self):
        with patch.dict(os.environ, {"LUMINA_RUNTIME_DIR": ""}, clear=False):
            self.assertEqual(runtime_root(), ROOT_DIR / "runtime")

    def test_runtime_root_supports_relative_env_override(self):
        with patch.dict(os.environ, {"LUMINA_RUNTIME_DIR": "tmp/runtime-custom"}, clear=False):
            self.assertEqual(runtime_root(), (ROOT_DIR / "tmp/runtime-custom").resolve())

    def test_backups_root_supports_relative_env_override(self):
        with patch.dict(os.environ, {"LUMINA_BACKUP_DIR": "tmp/backups-custom"}, clear=False):
            self.assertEqual(backups_root(), (ROOT_DIR / "tmp/backups-custom").resolve())


if __name__ == "__main__":
    unittest.main()
