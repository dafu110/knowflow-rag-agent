from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable

from .models import Chunk, Document


SUPPORTED_EXTENSIONS = {".txt", ".md", ".csv", ".json"}


def stable_id(*parts: str) -> str:
    digest = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
    return digest[:16]


def parse_document(path: Path, text: str | None = None) -> Document:
    raw = text if text is not None else path.read_text(encoding="utf-8", errors="ignore")
    metadata, body = _extract_front_matter(raw)
    title = str(metadata.get("title") or _guess_title(path, body))
    allowed_roles = _csv_set(metadata.get("allowed_roles"))
    allowed_users = _csv_set(metadata.get("allowed_users"))
    doc_id = stable_id(str(path), title, body[:2048])
    return Document(
        id=doc_id,
        title=title,
        source=str(path),
        text=body.strip(),
        metadata=metadata,
        allowed_roles=allowed_roles,
        allowed_users=allowed_users,
    )


def chunk_document(document: Document, max_chars: int = 900, overlap_chars: int = 120) -> list[Chunk]:
    sections = list(_split_sections(document.text))
    chunks: list[Chunk] = []
    ordinal = 0
    for section_path, section_text in sections:
        for piece in _split_piece(section_text, max_chars=max_chars, overlap_chars=overlap_chars):
            chunk_id = f"{document.id}:{ordinal:04d}"
            chunks.append(
                Chunk(
                    id=chunk_id,
                    document_id=document.id,
                    title=document.title,
                    source=document.source,
                    text=piece,
                    section_path=section_path,
                    metadata=dict(document.metadata),
                    allowed_roles=set(document.allowed_roles),
                    allowed_users=set(document.allowed_users),
                    ordinal=ordinal,
                    created_at=document.created_at,
                )
            )
            ordinal += 1
    return chunks


def load_documents_from_path(path: Path) -> list[Document]:
    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return []
        return [parse_document(path)]
    documents: list[Document] = []
    for file_path in sorted(path.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            documents.append(parse_document(file_path))
    return documents


def _extract_front_matter(raw: str) -> tuple[dict[str, str], str]:
    if not raw.startswith("---"):
        return {}, raw
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", raw, flags=re.DOTALL)
    if not match:
        return {}, raw
    metadata: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip()
    return metadata, match.group(2)


def _guess_title(path: Path, body: str) -> str:
    for line in body.splitlines():
        stripped = line.strip(" #\t")
        if stripped:
            return stripped[:120]
    return path.stem


def _csv_set(value: object) -> set[str]:
    if not value:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {str(item).strip() for item in value if str(item).strip()}
    return {part.strip() for part in str(value).split(",") if part.strip()}


def _split_sections(text: str) -> Iterable[tuple[list[str], str]]:
    current_path: list[str] = []
    current_lines: list[str] = []
    found_heading = False
    for line in text.splitlines():
        heading = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if heading:
            if current_lines:
                yield current_path[:], "\n".join(current_lines).strip()
                current_lines = []
            level = len(heading.group(1))
            title = heading.group(2).strip()
            current_path = current_path[: level - 1] + [title]
            found_heading = True
            current_lines.append(title)
        else:
            current_lines.append(line)
    if current_lines:
        yield current_path[:] if found_heading else [], "\n".join(current_lines).strip()


def _split_piece(text: str, max_chars: int, overlap_chars: int) -> Iterable[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return
    buffer = ""
    for paragraph in paragraphs:
        candidate = f"{buffer}\n\n{paragraph}".strip() if buffer else paragraph
        if len(candidate) <= max_chars:
            buffer = candidate
            continue
        if buffer:
            yield buffer
            tail = buffer[-overlap_chars:].strip()
            buffer = f"{tail}\n\n{paragraph}".strip() if tail else paragraph
        while len(buffer) > max_chars:
            yield buffer[:max_chars].strip()
            buffer = buffer[max(0, max_chars - overlap_chars) :].strip()
    if buffer:
        yield buffer

