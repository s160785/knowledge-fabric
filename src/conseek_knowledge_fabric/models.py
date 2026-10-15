"""Canonical, source-grounded knowledge graph models."""

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class SourceLocation:
	path: str
	line: int
	column: int
	end_line: int
	end_column: int


@dataclass(frozen=True)
class Symbol:
	id: str
	name: str
	kind: str
	language: str
	location: SourceLocation
	evidence: str
	source: str = "tree-sitter"
	symbol_summary: str | None = None


@dataclass(frozen=True)
class Relationship:
	source: str
	target: str
	kind: str
	location: SourceLocation
	evidence: str
	source_index: str = "tree-sitter"
	relationship_summary: str | None = None


@dataclass
class KnowledgeGraph:
	repository: str
	files: list[str] = field(default_factory=list)
	symbols: list[Symbol] = field(default_factory=list)
	relationships: list[Relationship] = field(default_factory=list)
	diagnostics: list[str] = field(default_factory=list)

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)