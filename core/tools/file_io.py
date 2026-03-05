import importlib
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from core.config import FileIOConfig
from core.paths import project_root, runtime_root
from core.tools.base import BaseTool
from core.tools.models import ToolContext, ToolResult


_PATH_LOCKS: Dict[str, threading.Lock] = {}
_PATH_LOCKS_GUARD = threading.Lock()


def _normalize_exts(values: Iterable[str]) -> Set[str]:
    exts: Set[str] = set()
    for raw in values:
        ext = str(raw).strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = f".{ext}"
        exts.add(ext)
    return exts


def _is_within(path: Path, root: Path) -> bool:
    try:
        common = os.path.commonpath([str(path), str(root)])
    except Exception:
        return False
    return os.path.normcase(common) == os.path.normcase(str(root))


def _resolve_root(raw: str) -> Path:
    text = str(raw or "").strip()
    token = text.upper()
    if token == "$PROJECT":
        return project_root().resolve()
    if token == "$RUNTIME":
        return runtime_root().resolve()
    root = Path(text).expanduser()
    if not root.is_absolute():
        root = project_root() / root
    return root.resolve()


@contextmanager
def _path_lock(path: Path):
    key = os.path.normcase(str(path))
    with _PATH_LOCKS_GUARD:
        lock = _PATH_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _PATH_LOCKS[key] = lock
    lock.acquire()
    try:
        yield
    finally:
        lock.release()


@dataclass
class _FilePolicy:
    allow_any_absolute_path: bool
    allowed_roots: List[Path]
    allowed_read_exts: Set[str]
    allowed_write_exts: Set[str]
    max_file_bytes: int
    max_chars: int
    max_pdf_pages: int
    default_encoding: str

    @classmethod
    def from_config(cls, cfg: FileIOConfig) -> "_FilePolicy":
        roots = [_resolve_root(raw) for raw in cfg.allowed_roots if str(raw or "").strip()]
        if not roots:
            roots = [project_root().resolve()]
        return cls(
            allow_any_absolute_path=bool(cfg.allow_any_absolute_path),
            allowed_roots=roots,
            allowed_read_exts=_normalize_exts(cfg.allowed_read_exts),
            allowed_write_exts=_normalize_exts(cfg.allowed_write_exts),
            max_file_bytes=max(int(cfg.max_file_bytes), 1),
            max_chars=max(int(cfg.max_chars), 1),
            max_pdf_pages=max(int(cfg.max_pdf_pages), 1),
            default_encoding=str(cfg.default_encoding or "utf-8").strip() or "utf-8",
        )

    def resolve_path(self, raw_path: str) -> Tuple[Optional[Path], Optional[str]]:
        path_text = str(raw_path or "").strip()
        if not path_text:
            return None, "missing `path`"

        is_absolute_path = False
        try:
            path = Path(path_text).expanduser()
            is_absolute_path = path.is_absolute()
            if not path.is_absolute():
                path = project_root() / path
            resolved = path.resolve()
        except Exception as exc:
            return None, f"invalid path: {exc}"

        if self.allow_any_absolute_path and is_absolute_path:
            return resolved, None
        if not any(_is_within(resolved, root) for root in self.allowed_roots):
            return None, "path is outside allowed roots"
        return resolved, None

    def check_ext(self, path: Path, *, write: bool, extra_allowed: Optional[Set[str]] = None) -> bool:
        ext = str(path.suffix or "").lower()
        if not ext:
            return False
        base = self.allowed_write_exts if write else self.allowed_read_exts
        if ext not in base:
            return False
        if extra_allowed is not None and ext not in extra_allowed:
            return False
        return True


class _FileToolBase(BaseTool):
    def __init__(
        self,
        *,
        config: FileIOConfig,
        name: str,
        description: str,
        parameters_schema: Dict[str, Any],
    ):
        self.policy = _FilePolicy.from_config(config)
        super().__init__(name=name, description=description, parameters_schema=parameters_schema)

    def _read_path(self, raw_path: str, *, only_exts: Optional[Set[str]] = None) -> Tuple[Optional[Path], Optional[ToolResult]]:
        path, err = self.policy.resolve_path(raw_path)
        if path is None:
            code = "FILE_ACCESS_DENIED" if err and "outside allowed roots" in err else "FILE_BAD_INPUT"
            return None, self.error_result(
                code=code,
                message=f"Invalid path: {err}",
                retryable=False,
            )
        if not self.policy.check_ext(path, write=False, extra_allowed=only_exts):
            return None, self.error_result(
                code="FILE_ACCESS_DENIED",
                message=f"Read extension not allowed: {path.suffix or '<none>'}",
                retryable=False,
            )
        return path, None

    def _write_path(self, raw_path: str, *, only_exts: Optional[Set[str]] = None) -> Tuple[Optional[Path], Optional[ToolResult]]:
        path, err = self.policy.resolve_path(raw_path)
        if path is None:
            code = "FILE_ACCESS_DENIED" if err and "outside allowed roots" in err else "FILE_BAD_INPUT"
            return None, self.error_result(
                code=code,
                message=f"Invalid path: {err}",
                retryable=False,
            )
        if not self.policy.check_ext(path, write=True, extra_allowed=only_exts):
            return None, self.error_result(
                code="FILE_ACCESS_DENIED",
                message=f"Write extension not allowed: {path.suffix or '<none>'}",
                retryable=False,
            )
        return path, None

    def _max_chars(self, value: Optional[int]) -> int:
        return self.clamp_int(value, default=self.policy.max_chars, min_value=1, max_value=self.policy.max_chars)

    def _truncate(self, text: str, limit: int) -> Tuple[str, bool]:
        if len(text) <= limit:
            return text, False
        return text[:limit], True

    def _file_size_or_zero(self, path: Path) -> int:
        try:
            return int(path.stat().st_size)
        except Exception:
            return 0

    def _enforce_size_limit(self, size_bytes: int) -> Optional[ToolResult]:
        if size_bytes <= self.policy.max_file_bytes:
            return None
        return self.error_result(
            code="FILE_TOO_LARGE",
            message=f"File exceeds size limit: {size_bytes} > {self.policy.max_file_bytes}",
            retryable=False,
        )


class ReadFileTool(_FileToolBase):
    def __init__(self, config: FileIOConfig):
        super().__init__(
            config=config,
            name="read_file",
            description="Read text files (including markdown) from allowed local paths.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Target file path."},
                    "start_line": {"type": "integer", "description": "1-based start line."},
                    "end_line": {"type": "integer", "description": "1-based end line."},
                    "max_chars": {"type": "integer", "description": "Max returned characters."},
                    "encoding": {"type": "string", "description": "Text encoding, default utf-8."},
                },
                "required": ["path"],
            },
        )

    def run(
        self,
        *,
        ctx: ToolContext,
        path: str,
        start_line: int = 1,
        end_line: Optional[int] = None,
        max_chars: Optional[int] = None,
        encoding: Optional[str] = None,
        **kwargs: Any,
    ) -> ToolResult:
        _ = ctx, kwargs
        resolved, err = self._read_path(path)
        if err is not None:
            return err
        if resolved is None:
            return self.error_result(code="FILE_IO_ERROR", message="Path resolve failed", retryable=False)
        if not resolved.exists():
            return self.error_result(code="FILE_NOT_FOUND", message=f"File not found: {resolved}", retryable=False)
        if not resolved.is_file():
            return self.error_result(code="FILE_BAD_INPUT", message=f"Not a file: {resolved}", retryable=False)

        size_guard = self._enforce_size_limit(self._file_size_or_zero(resolved))
        if size_guard is not None:
            return size_guard

        text_encoding = str(encoding or self.policy.default_encoding).strip() or self.policy.default_encoding
        try:
            with open(resolved, "r", encoding=text_encoding) as f:
                raw_text = f.read()
        except UnicodeDecodeError:
            return self.error_result(
                code="FILE_NOT_TEXT",
                message=f"Unable to decode text file with encoding={text_encoding}",
                retryable=False,
            )
        except OSError as exc:
            return self.error_result(
                code="FILE_IO_ERROR",
                message=f"Read file failed: {exc}",
                retryable=True,
            )

        body, effective_start, effective_end, total_lines = self._slice_lines(raw_text, start_line, end_line)
        content, truncated = self._truncate(body, self._max_chars(max_chars))
        return self.ok_result(
            {
                "path": str(resolved),
                "encoding": text_encoding,
                "line_start": effective_start,
                "line_end": effective_end,
                "total_lines": total_lines,
                "truncated": truncated,
                "content": content,
            }
        )

    def _slice_lines(
        self,
        raw_text: str,
        start_line: int,
        end_line: Optional[int],
    ) -> Tuple[str, int, int, int]:
        lines = raw_text.splitlines()
        total_lines = len(lines)
        try:
            start = int(start_line)
        except Exception:
            start = 1
        start = max(start, 1)

        if end_line is None:
            end = total_lines
        else:
            try:
                end = int(end_line)
            except Exception:
                end = start
            end = max(end, start)

        start_idx = min(max(start - 1, 0), total_lines)
        end_idx = min(max(end, 0), total_lines)
        if end_idx < start_idx:
            end_idx = start_idx

        selected_lines = lines[start_idx:end_idx]
        content = "\n".join(selected_lines)
        if selected_lines:
            effective_start = start_idx + 1
            effective_end = start_idx + len(selected_lines)
        else:
            effective_start = start
            effective_end = start - 1
        return content, effective_start, effective_end, total_lines


class ReadPdfTool(_FileToolBase):
    def __init__(self, config: FileIOConfig):
        super().__init__(
            config=config,
            name="read_pdf",
            description="Read text from PDF files in allowed local paths.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Target PDF path."},
                    "page_start": {"type": "integer", "description": "1-based start page."},
                    "page_end": {"type": "integer", "description": "1-based end page."},
                    "max_chars": {"type": "integer", "description": "Max returned characters."},
                },
                "required": ["path"],
            },
        )

    def run(
        self,
        *,
        ctx: ToolContext,
        path: str,
        page_start: int = 1,
        page_end: Optional[int] = None,
        max_chars: Optional[int] = None,
        **kwargs: Any,
    ) -> ToolResult:
        _ = ctx, kwargs
        resolved, err = self._read_path(path, only_exts={".pdf"})
        if err is not None:
            return err
        if resolved is None:
            return self.error_result(code="FILE_IO_ERROR", message="Path resolve failed", retryable=False)
        if not resolved.exists():
            return self.error_result(code="FILE_NOT_FOUND", message=f"File not found: {resolved}", retryable=False)
        if not resolved.is_file():
            return self.error_result(code="FILE_BAD_INPUT", message=f"Not a file: {resolved}", retryable=False)

        size_guard = self._enforce_size_limit(self._file_size_or_zero(resolved))
        if size_guard is not None:
            return size_guard

        try:
            pypdf = importlib.import_module("pypdf")
            pdf_reader_cls = getattr(pypdf, "PdfReader")
        except Exception:
            return self.error_result(
                code="FILE_DEPENDENCY_MISSING",
                message="Missing dependency for PDF reading: pypdf",
                retryable=False,
            )

        try:
            reader = pdf_reader_cls(str(resolved))
        except Exception as exc:
            return self.error_result(
                code="FILE_IO_ERROR",
                message=f"Open PDF failed: {exc}",
                retryable=True,
            )

        if bool(getattr(reader, "is_encrypted", False)):
            try:
                decrypt_result = int(reader.decrypt("") or 0)
            except Exception:
                decrypt_result = 0
            if decrypt_result <= 0:
                return self.error_result(
                    code="FILE_PDF_ENCRYPTED",
                    message="PDF is encrypted and cannot be read without password",
                    retryable=False,
                )

        total_pages = int(len(reader.pages))
        if total_pages <= 0:
            return self.error_result(
                code="FILE_PDF_NO_TEXT",
                message="PDF has no readable pages",
                retryable=False,
            )

        start = self.clamp_int(page_start, default=1, min_value=1, max_value=total_pages)
        if page_end is None:
            end = total_pages
        else:
            end = self.clamp_int(page_end, default=start, min_value=1, max_value=total_pages)
        if end < start:
            return self.error_result(
                code="FILE_BAD_INPUT",
                message=f"Invalid page range: start={start}, end={end}",
                retryable=False,
            )
        if (end - start + 1) > self.policy.max_pdf_pages:
            return self.error_result(
                code="FILE_BAD_INPUT",
                message=(
                    f"Requested page range too large: {end - start + 1} > {self.policy.max_pdf_pages}. "
                    "Please narrow page_start/page_end."
                ),
                retryable=False,
            )

        chunks: List[str] = []
        for page_num in range(start, end + 1):
            page = reader.pages[page_num - 1]
            try:
                text = str(page.extract_text() or "").strip()
            except Exception:
                text = ""
            if text:
                chunks.append(f"[page {page_num}]\n{text}")

        content_text = "\n\n".join(chunks).strip()
        if not content_text:
            return self.error_result(
                code="FILE_PDF_NO_TEXT",
                message="No extractable text found in requested PDF pages",
                retryable=False,
            )

        content, truncated = self._truncate(content_text, self._max_chars(max_chars))
        return self.ok_result(
            {
                "path": str(resolved),
                "page_start": start,
                "page_end": end,
                "total_pages": total_pages,
                "extracted_pages": end - start + 1,
                "truncated": truncated,
                "content": content,
            }
        )


class WriteMarkdownTool(_FileToolBase):
    _SUPPORTED_MODES = {"overwrite", "append", "append_section"}

    def __init__(self, config: FileIOConfig):
        super().__init__(
            config=config,
            name="write_markdown",
            description="Write markdown files in allowed local paths.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Target markdown path."},
                    "content": {"type": "string", "description": "Content to write."},
                    "mode": {
                        "type": "string",
                        "description": "Write mode: overwrite | append | append_section.",
                    },
                    "section_title": {
                        "type": "string",
                        "description": "Required when mode=append_section.",
                    },
                    "encoding": {"type": "string", "description": "Output encoding, default utf-8."},
                },
                "required": ["path", "content"],
            },
        )

    def run(
        self,
        *,
        ctx: ToolContext,
        path: str,
        content: str,
        mode: str = "overwrite",
        section_title: Optional[str] = None,
        encoding: Optional[str] = None,
        **kwargs: Any,
    ) -> ToolResult:
        _ = ctx, kwargs
        resolved, err = self._write_path(path, only_exts={".md"})
        if err is not None:
            return err
        if resolved is None:
            return self.error_result(code="FILE_IO_ERROR", message="Path resolve failed", retryable=False)

        mode_text = str(mode or "overwrite").strip().lower()
        if mode_text not in self._SUPPORTED_MODES:
            return self.error_result(
                code="FILE_BAD_INPUT",
                message=f"Unsupported write mode: {mode_text}",
                retryable=False,
            )

        text_encoding = str(encoding or self.policy.default_encoding).strip() or self.policy.default_encoding
        content_text = str(content or "")
        title_text = str(section_title or "").strip()
        if mode_text == "append_section" and not title_text:
            return self.error_result(
                code="FILE_BAD_INPUT",
                message="section_title is required when mode=append_section",
                retryable=False,
            )

        bytes_written = 0
        created = False
        try:
            with _path_lock(resolved):
                resolved.parent.mkdir(parents=True, exist_ok=True)
                existed_before = resolved.exists()
                created = not existed_before

                if mode_text == "overwrite":
                    final_text = content_text
                    bytes_written = len(final_text.encode(text_encoding, errors="ignore"))
                    size_guard = self._enforce_size_limit(bytes_written)
                    if size_guard is not None:
                        return size_guard
                    self._atomic_write(resolved, final_text, encoding=text_encoding)
                elif mode_text == "append":
                    final_text = content_text
                    bytes_written = len(final_text.encode(text_encoding, errors="ignore"))
                    new_size = self._file_size_or_zero(resolved) + bytes_written
                    size_guard = self._enforce_size_limit(new_size)
                    if size_guard is not None:
                        return size_guard
                    with open(resolved, "a", encoding=text_encoding) as f:
                        f.write(final_text)
                else:
                    final_text = f"\n\n## {title_text}\n\n{content_text.strip()}\n"
                    bytes_written = len(final_text.encode(text_encoding, errors="ignore"))
                    new_size = self._file_size_or_zero(resolved) + bytes_written
                    size_guard = self._enforce_size_limit(new_size)
                    if size_guard is not None:
                        return size_guard
                    with open(resolved, "a", encoding=text_encoding) as f:
                        f.write(final_text)
        except OSError as exc:
            return self.error_result(
                code="FILE_IO_ERROR",
                message=f"Write markdown failed: {exc}",
                retryable=True,
            )

        return self.ok_result(
            {
                "path": str(resolved),
                "mode": mode_text,
                "bytes_written": int(bytes_written),
                "created": bool(created),
            }
        )

    def _atomic_write(self, path: Path, content: str, *, encoding: str) -> None:
        temp_path = path.with_name(f"{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
        try:
            with open(temp_path, "w", encoding=encoding) as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())
            temp_path.replace(path)
        finally:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass
