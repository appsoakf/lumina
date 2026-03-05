import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_pet_e2e import prepare_fresh_runtime


class RunPetE2ETests(unittest.TestCase):
    def test_prepare_fresh_runtime_recreates_existing_directory(self):
        with tempfile.TemporaryDirectory(prefix="lumina-e2e-runtime-") as tmp:
            root = Path(tmp) / "repo"
            runtime_dir = root / "runtime" / "e2e" / "current"
            stale_file = runtime_dir / "logs" / "old.log"
            stale_file.parent.mkdir(parents=True, exist_ok=True)
            stale_file.write_text("stale", encoding="utf-8")

            prepared = prepare_fresh_runtime(runtime_dir, project_root_dir=root)

            self.assertEqual(prepared, runtime_dir.resolve())
            self.assertTrue(runtime_dir.exists())
            self.assertFalse(stale_file.exists())

    def test_prepare_fresh_runtime_rejects_path_outside_project_root(self):
        with tempfile.TemporaryDirectory(prefix="lumina-e2e-runtime-") as tmp:
            root = Path(tmp) / "repo"
            outside_runtime = Path(tmp) / "outside-runtime"
            root.mkdir(parents=True, exist_ok=True)

            with self.assertRaises(ValueError):
                prepare_fresh_runtime(outside_runtime, project_root_dir=root)


if __name__ == "__main__":
    unittest.main()
