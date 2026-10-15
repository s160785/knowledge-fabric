"""Enrich syntax extraction with references from an official SCIP index."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from conseek_knowledge_fabric.models import KnowledgeGraph, Relationship, SourceLocation, Symbol


def _range(occurrence: dict[str, Any], path: str) -> SourceLocation | None:
    single = occurrence.get("singleLineRange") or occurrence.get("singleLineEnclosingRange")
    multi = occurrence.get("multiLineRange") or occurrence.get("multiLineEnclosingRange")
    legacy = occurrence.get("range")
    if single:
        line = int(single.get("line", 0))
        start = int(single.get("startCharacter", 0))
        end_line = line
        end = int(single.get("endCharacter", start))
    elif multi:
        line = int(multi.get("startLine", 0))
        start = int(multi.get("startCharacter", 0))
        end_line = int(multi.get("endLine", line))
        end = int(multi.get("endCharacter", start))
    elif legacy and len(legacy) >= 3:
        line, start = int(legacy[0]), int(legacy[1])
        if len(legacy) == 3:
            end_line, end = line, int(legacy[2])
        else:
            end_line, end = int(legacy[2]), int(legacy[3])
    else:
        return None
    return SourceLocation(path, line + 1, start, end_line + 1, end)


def _owner(graph: KnowledgeGraph, location: SourceLocation) -> str:
    matches = [
        symbol for symbol in graph.symbols
        if symbol.location.path == location.path
        and symbol.location.line <= location.line <= symbol.location.end_line
    ]
    if not matches:
        return f"{location.path}::module"
    return min(matches, key=lambda symbol: symbol.location.end_line - symbol.location.line).id


def _project_path(value: str, fallback: str) -> Path:
    parsed = urlsplit(value or fallback)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path)).resolve()
    return Path(unquote(value or fallback)).resolve()


def enrich_from_scip(
    graph: KnowledgeGraph,
    index_path: str | Path,
    command: str = "scip",
    timeout: int = 120,
) -> None:
    """Merge definitions and references from `scip print --json` into graph."""
    executable = shutil.which(command)
    if executable is None:
        raise FileNotFoundError(f"SCIP CLI not found: {command}")
    result = subprocess.run(
        [executable, "print", "--json", str(Path(index_path).resolve())],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or f"scip exited with status {result.returncode}")
    index = json.loads(result.stdout)
    metadata = index.get("metadata", {})
    project_root = _project_path(metadata.get("projectRoot", ""), graph.repository)
    symbol_ids: dict[str, str] = {}
    definitions: list[tuple[str, str, SourceLocation, str, str]] = []
    references: list[tuple[str, SourceLocation]] = []

    for document in index.get("documents", []):
        absolute_path = (project_root / document.get("relativePath", "")).resolve()
        try:
            relative_path = absolute_path.relative_to(Path(graph.repository)).as_posix()
        except ValueError:
            graph.diagnostics.append(f"Ignored SCIP document outside repository: {document.get('relativePath', '')}")
            continue
        document_symbols = {item.get("symbol", ""): item for item in document.get("symbols", [])}
        occurrences = document.get("occurrences", [])
        for occurrence in occurrences:
            scip_id = occurrence.get("symbol")
            location = _range(occurrence, relative_path)
            if not scip_id or location is None:
                continue
            if int(occurrence.get("symbolRoles", 0)) & 1:
                info = document_symbols.get(scip_id, {})
                name = info.get("displayName") or scip_id
                definitions.append((scip_id, name, location, str(info.get("kind", "symbol")), "\n".join(info.get("documentation", []))))
            else:
                references.append((scip_id, location))

    for scip_id, name, location, kind, documentation in definitions:
        existing = next((
            symbol for symbol in graph.symbols
            if symbol.location.path == location.path
            and symbol.location.line == location.line
            and symbol.name == name
        ), None)
        if existing:
            symbol_ids[scip_id] = existing.id
            continue
        identifier = f"scip:{scip_id}"
        symbol_ids[scip_id] = identifier
        graph.symbols.append(Symbol(
            id=identifier,
            name=name,
            kind=kind,
            language="unknown",
            location=location,
            evidence=documentation,
            source="scip",
        ))

    for scip_id, location in references:
        graph.relationships.append(Relationship(
            source=_owner(graph, location),
            target=symbol_ids.get(scip_id, scip_id),
            kind="references",
            location=location,
            evidence=scip_id,
            source_index="scip",
        ))