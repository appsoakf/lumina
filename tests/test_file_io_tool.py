import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import FileIOConfig
from core.tools import ToolContext, build_default_registry
from core.tools.file_io import ReadFileTool, ReadPdfTool, WriteMarkdownTool


class FileIOToolTests(unittest.TestCase):
    def _cfg(
        self,
        *,
        roots,
        allow_any_absolute_path=False,
        max_file_bytes=2 * 1024 * 1024,
        max_chars=12000,
        max_pdf_pages=20,
    ) -> FileIOConfig:
        return FileIOConfig(
            enabled=True,
            allow_any_absolute_path=allow_any_absolute_path,
            allowed_roots=list(roots),
            allowed_read_exts=[".txt", ".md", ".pdf"],
            allowed_write_exts=[".md"],
            max_file_bytes=max_file_bytes,
            max_chars=max_chars,
            max_pdf_pages=max_pdf_pages,
            default_encoding="utf-8",
        )

    def test_build_default_registry_contains_file_io_tools(self):
        registry = build_default_registry()
        names = [item["function"]["name"] for item in registry.list_schemas()]
        self.assertIn("read_file", names)
        self.assertIn("read_pdf", names)
        self.assertIn("write_markdown", names)

    def test_read_file_reads_markdown_line_range(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            md_path = root / "demo.md"
            md_path.write_text("line1\nline2\nline3\nline4\n", encoding="utf-8")

            tool = ReadFileTool(config=self._cfg(roots=[str(root)]))
            result = tool.run(
                ctx=ToolContext(session_id="s1"),
                path=str(md_path),
                start_line=2,
                end_line=3,
                max_chars=100,
            )

            self.assertTrue(result.ok)
            body = json.loads(result.content)
            self.assertEqual(body["line_start"], 2)
            self.assertEqual(body["line_end"], 3)
            self.assertEqual(body["content"], "line2\nline3")
            self.assertFalse(body["truncated"])

    def test_read_file_rejects_path_outside_allowed_root(self):
        with tempfile.TemporaryDirectory() as allowed_dir, tempfile.TemporaryDirectory() as outside_dir:
            outside_path = Path(outside_dir) / "outside.md"
            outside_path.write_text("blocked", encoding="utf-8")

            tool = ReadFileTool(config=self._cfg(roots=[allowed_dir]))
            result = tool.run(ctx=ToolContext(session_id="s1"), path=str(outside_path))

            self.assertFalse(result.ok)
            body = json.loads(result.content)
            self.assertEqual(body["error_code"], "FILE_ACCESS_DENIED")
            self.assertIn("outside allowed roots", body["message"])

    def test_read_file_allows_absolute_path_when_enabled(self):
        with tempfile.TemporaryDirectory() as allowed_dir, tempfile.TemporaryDirectory() as outside_dir:
            outside_path = Path(outside_dir) / "outside.md"
            outside_path.write_text("allowed", encoding="utf-8")

            tool = ReadFileTool(config=self._cfg(roots=[allowed_dir], allow_any_absolute_path=True))
            result = tool.run(ctx=ToolContext(session_id="s1"), path=str(outside_path))

            self.assertTrue(result.ok)
            body = json.loads(result.content)
            self.assertEqual(body["content"], "allowed")

    def test_write_markdown_overwrite_then_append_section(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            path = root / "article.md"
            tool = WriteMarkdownTool(config=self._cfg(roots=[str(root)]))

            first = tool.run(
                ctx=ToolContext(session_id="s1"),
                path=str(path),
                content="# 标题\n第一段",
                mode="overwrite",
            )
            self.assertTrue(first.ok)
            first_body = json.loads(first.content)
            self.assertTrue(first_body["created"])

            second = tool.run(
                ctx=ToolContext(session_id="s1"),
                path=str(path),
                content="这是续写内容。",
                mode="append_section",
                section_title="补充",
            )
            self.assertTrue(second.ok)
            second_body = json.loads(second.content)
            self.assertFalse(second_body["created"])

            final_text = path.read_text(encoding="utf-8")
            self.assertIn("# 标题", final_text)
            self.assertIn("## 补充", final_text)
            self.assertIn("这是续写内容。", final_text)

    def test_write_markdown_rejects_non_markdown_extension(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            tool = WriteMarkdownTool(config=self._cfg(roots=[str(root)]))
            result = tool.run(
                ctx=ToolContext(session_id="s1"),
                path=str(root / "article.txt"),
                content="x",
                mode="overwrite",
            )

            self.assertFalse(result.ok)
            body = json.loads(result.content)
            self.assertEqual(body["error_code"], "FILE_ACCESS_DENIED")

    def test_write_markdown_allows_absolute_path_when_enabled(self):
        with tempfile.TemporaryDirectory() as allowed_dir, tempfile.TemporaryDirectory() as outside_dir:
            outside_path = Path(outside_dir) / "article.md"
            tool = WriteMarkdownTool(config=self._cfg(roots=[allowed_dir], allow_any_absolute_path=True))

            result = tool.run(
                ctx=ToolContext(session_id="s1"),
                path=str(outside_path),
                content="hello",
                mode="overwrite",
            )

            self.assertTrue(result.ok)
            self.assertTrue(outside_path.exists())
            self.assertEqual(outside_path.read_text(encoding="utf-8"), "hello")

    def test_read_pdf_returns_dependency_error_when_pypdf_missing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            pdf_path = root / "demo.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\n")
            tool = ReadPdfTool(config=self._cfg(roots=[str(root)]))

            with mock.patch("core.tools.file_io.importlib.import_module", side_effect=ImportError("missing")):
                result = tool.run(ctx=ToolContext(session_id="s1"), path=str(pdf_path))

            self.assertFalse(result.ok)
            body = json.loads(result.content)
            self.assertEqual(body["error_code"], "FILE_DEPENDENCY_MISSING")


if __name__ == "__main__":
    unittest.main()
