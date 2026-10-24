"""Persist a Tree-sitter knowledge graph into Neo4j."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Mapping

from neo4j import Driver
from neo4j.exceptions import Neo4jError

from conseek_knowledge_fabric.models import KnowledgeGraph, Relationship, SourceLocation, Symbol
from conseek_knowledge_fabric.utils.logging import get_logger, log_call


logger = get_logger(__name__)


@log_call(logger)
def _groups_by_file(
	graph: KnowledgeGraph,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
	local_ids = {symbol.id for symbol in graph.symbols}
	symbols: dict[str, list[dict[str, Any]]] = defaultdict(list)
	for symbol in graph.symbols:
		symbols[symbol.location.path].append({
			"id": f"{graph.repository}::{symbol.id}",
			"name": symbol.name,
			"kind": symbol.kind,
			"language": symbol.language,
			"source": symbol.source,
			"evidence": symbol.evidence,
			"line": symbol.location.line,
			"column": symbol.location.column,
			"end_line": symbol.location.end_line,
			"end_column": symbol.location.end_column,
			"symbol_summary": symbol.symbol_summary,
			"symbol_summary_embedding": symbol.symbol_summary_embedding,
		})
	relationships: dict[str, list[dict[str, Any]]] = defaultdict(list)
	for relationship in graph.relationships:
		target_id = (
			f"{graph.repository}::{relationship.target}"
			if relationship.target in local_ids
			else f"{graph.repository}::external::{relationship.target}"
		)
		relationships[relationship.location.path].append({
			"source": f"{graph.repository}::{relationship.source}" if relationship.source in local_ids else None,
			"target": target_id,
			"local_target": relationship.target in local_ids,
			"kind": relationship.kind,
			"source_index": relationship.source_index,
			"evidence": relationship.evidence,
			"line": relationship.location.line,
			"column": relationship.location.column,
			"end_line": relationship.location.end_line,
			"end_column": relationship.location.end_column,
			"relationship_summary": relationship.relationship_summary,
			"relationship_summary_embedding": relationship.relationship_summary_embedding,
		})
	return symbols, relationships


@log_call(logger)
def _write_file(tx: Any, repository: str, path: str, symbols: list[dict[str, Any]], relationships: list[dict[str, Any]]) -> None:
	tx.run(
		"""
		MERGE (repo:Repository {path: $repository})
		MERGE (file:File {path: $path, repository: $repository})
		SET file.name = $name
		MERGE (repo)-[:CONTAINS]->(file)
		WITH repo, file
		UNWIND $symbols AS item
		MERGE (symbol:Symbol {id: item.id})
		SET symbol.name = item.name, symbol.kind = item.kind,
		    symbol.language = item.language, symbol.source = item.source,
		    symbol.evidence = item.evidence, symbol.line = item.line,
		    symbol.column = item.column, symbol.end_line = item.end_line,
		    symbol.end_column = item.end_column,
		    symbol.symbol_summary = item.symbol_summary,
		    symbol.symbol_summary_embedding = item.symbol_summary_embedding
		MERGE (file)-[:DEFINES]->(symbol)
		""",
		repository=repository,
		path=path,
		name=path.rsplit("/", 1)[-1],
		symbols=symbols,
	).consume()
	if not relationships:
		return
	tx.run(
		"""
		UNWIND $relationships AS item
		MATCH (repo:Repository {path: $repository})
		MATCH (file:File {path: $path, repository: $repository})
		WITH repo, file, item
		OPTIONAL MATCH (source:Symbol {id: item.source})
		WITH repo, file, item, source
		CALL {
			WITH item
			WITH item WHERE item.kind = 'imports'
			MERGE (target:Import {id: item.target, repository: $repository})
			SET target.statement = item.evidence
			RETURN target
			UNION
			WITH item
			WITH item WHERE item.kind <> 'imports' AND item.local_target
			MATCH (target:Symbol {id: item.target})
			RETURN target
			UNION
			WITH item
			WITH item WHERE item.kind <> 'imports' AND NOT item.local_target
			MERGE (target:ExternalSymbol {id: item.target, repository: $repository})
			SET target.name = item.target
			RETURN target
		}
		MERGE (file)-[edge:CODE_RELATION {kind: item.kind, line: item.line, column: item.column, evidence: item.evidence}]->(target)
		SET edge.source = item.source,
		    edge.source_index = item.source_index,
		    edge.end_line = item.end_line, edge.end_column = item.end_column,
		    edge.summary = item.relationship_summary,
		    edge.relationship_summary = item.relationship_summary,
		    edge.relationship_summary_embedding = item.relationship_summary_embedding
		FOREACH (_ IN CASE WHEN source IS NULL THEN [] ELSE [1] END | MERGE (source)-[:CODE_RELATION {kind: item.kind, line: item.line, column: item.column, evidence: item.evidence}]->(target))
		""",
		repository=repository,
		path=path,
		relationships=relationships,
	).consume()


@log_call(logger)
def persist_graph(driver: Driver, graph: KnowledgeGraph, database: str | None = None) -> dict[str, int]:
	"""Upsert graph nodes and edges; stable IDs make repeated imports idempotent."""
	repository = graph.repository
	symbols_by_file, relationships_by_file = _groups_by_file(graph)
	files = sorted(set(graph.files) | set(symbols_by_file) | set(relationships_by_file))
	with driver.session(database=database) as session:
		logger.info("Creating Neo4j graph constraints")
		session.run("MERGE (:Repository {path: $repository})", repository=repository).consume()
		session.run("CREATE CONSTRAINT repository_path IF NOT EXISTS FOR (n:Repository) REQUIRE n.path IS UNIQUE").consume()
		session.run("CREATE CONSTRAINT file_identity IF NOT EXISTS FOR (n:File) REQUIRE (n.repository, n.path) IS UNIQUE").consume()
		session.run("CREATE CONSTRAINT symbol_id IF NOT EXISTS FOR (n:Symbol) REQUIRE n.id IS UNIQUE").consume()
		for path in files:
			logger.info("Persisting file graph")
			session.execute_write(
				_write_file,
				repository,
				path,
				symbols_by_file.get(path, []),
				relationships_by_file.get(path, []),
			)
	return {"files": len(files), "symbols": len(graph.symbols), "relationships": len(graph.relationships)}


@log_call(logger)
def store_graph(
	graph: KnowledgeGraph,
	uri: str,
	username: str,
	password: str,
	database: str | None = None,
) -> dict[str, int]:
	"""Connect, validate Neo4j connectivity, and persist the extracted graph."""
	from neo4j import GraphDatabase

	try:
		logger.info("Connecting to Neo4j")
		with GraphDatabase.driver(uri, auth=(username, password)) as driver:
			driver.verify_connectivity()
			logger.info("Neo4j connectivity verified")
			return persist_graph(driver, graph, database)
	except Neo4jError as error:
		logger.error("Neo4j operation failed exception_type=%s", type(error).__name__)
		raise RuntimeError("Neo4j operation failed; check its URI, credentials, database, and permissions") from error


def _strip_repo_prefix(value: str | None, repository: str) -> str:
	if not value:
		return "unknown"
	for prefix in (f"{repository}::external::", f"{repository}::"):
		if value.startswith(prefix):
			return value[len(prefix):]
	return value


@log_call(logger)
def neo4j_to_json(
	uri: str,
	username: str,
	password: str,
	repository: str | None = None,
	database: str | None = None,
	output_path: str | Path | None = None,
) -> dict[str, Any]:
	"""Export the existing Neo4j knowledge graph into canonical KnowledgeGraph JSON format."""
	from neo4j import GraphDatabase

	try:
		with GraphDatabase.driver(uri, auth=(username, password)) as driver:
			driver.verify_connectivity()
			with driver.session(database=database) as session:
				if not repository:
					rec = session.run("MATCH (r:Repository) RETURN r.path AS path LIMIT 1").single()
					repository = rec["path"] if rec else ""

				files = [
					r["path"]
					for r in session.run(
						"MATCH (f:File {repository: $repo}) RETURN f.path AS path ORDER BY f.path",
						repo=repository,
					)
				]
				symbols = []
				for r in session.run(
					"""
					MATCH (f:File {repository: $repo})-[:DEFINES]->(s:Symbol)
					RETURN f.path AS path, properties(s) AS props
					""",
					repo=repository,
				):
					p = r["props"]
					symbols.append({
						"id": _strip_repo_prefix(p.get("id"), repository),
						"name": p.get("name", ""),
						"kind": p.get("kind", ""),
						"language": p.get("language", ""),
						"location": {
							"path": r["path"],
							"line": int(p.get("line", 1)),
							"column": int(p.get("column", 0)),
							"end_line": int(p.get("end_line", 1)),
							"end_column": int(p.get("end_column", 0)),
						},
						"evidence": p.get("evidence", ""),
						"source": p.get("source", "tree-sitter"),
						"symbol_summary": p.get("symbol_summary"),
						"symbol_summary_embedding": p.get("symbol_summary_embedding"),
					})

				relationships = []
				for r in session.run(
					"""
					MATCH (f:File {repository: $repo})-[e:CODE_RELATION]->(t)
					OPTIONAL MATCH (f)-[:DEFINES]->(s:Symbol)-[:CODE_RELATION {
						kind: e.kind, line: e.line, column: e.column, evidence: e.evidence
					}]->(t)
					RETURN f.path AS path, coalesce(e.source, s.id) AS source, t.id AS target, properties(e) AS props
					""",
					repo=repository,
				):
					p = r["props"]
					relationships.append({
						"source": _strip_repo_prefix(r["source"], repository),
						"target": _strip_repo_prefix(r["target"], repository),
						"kind": p.get("kind", "related_to"),
						"location": {
							"path": r["path"],
							"line": int(p.get("line", 1)),
							"column": int(p.get("column", 0)),
							"end_line": int(p.get("end_line", 1)),
							"end_column": int(p.get("end_column", 0)),
						},
						"evidence": p.get("evidence", ""),
						"source_index": p.get("source_index", "tree-sitter"),
						"relationship_summary": p.get("relationship_summary") or p.get("summary"),
						"relationship_summary_embedding": p.get("relationship_summary_embedding"),
					})

				data = {
					"repository": repository,
					"files": files,
					"symbols": symbols,
					"relationships": relationships,
					"diagnostics": [],
				}
				if output_path:
					Path(output_path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
				return data
	except Neo4jError as error:
		logger.error("Neo4j export failed exception_type=%s", type(error).__name__)
		raise RuntimeError("Neo4j export failed; check its URI, credentials, database, and permissions") from error


def _symbol_key(sym: Mapping[str, Any] | Symbol) -> tuple[Any, ...]:
	if isinstance(sym, Symbol):
		loc = sym.location
		return (sym.id, sym.name, sym.kind, sym.language, loc.path, loc.line, loc.column, loc.end_line, loc.end_column)
	loc = sym.get("location") or {}
	return (
		sym.get("id"),
		sym.get("name"),
		sym.get("kind"),
		sym.get("language"),
		loc.get("path"),
		int(loc.get("line", 1)),
		int(loc.get("column", 0)),
		int(loc.get("end_line", 1)),
		int(loc.get("end_column", 0)),
	)


def _rel_key(rel: Mapping[str, Any] | Relationship) -> tuple[Any, ...]:
	if isinstance(rel, Relationship):
		loc = rel.location
		return (rel.source, rel.target, rel.kind, loc.path, loc.line, loc.column, loc.end_line, loc.end_column)
	loc = rel.get("location") or {}
	return (
		rel.get("source"),
		rel.get("target"),
		rel.get("kind"),
		loc.get("path"),
		int(loc.get("line", 1)),
		int(loc.get("column", 0)),
		int(loc.get("end_line", 1)),
		int(loc.get("end_column", 0)),
	)


def _dict_to_symbol(item: Mapping[str, Any] | Symbol) -> Symbol:
	if isinstance(item, Symbol):
		return item
	loc = item.get("location") or {}
	return Symbol(
		id=str(item["id"]),
		name=str(item.get("name", "")),
		kind=str(item.get("kind", "")),
		language=str(item.get("language", "")),
		location=SourceLocation(
			path=str(loc.get("path", "")),
			line=int(loc.get("line", 1)),
			column=int(loc.get("column", 0)),
			end_line=int(loc.get("end_line", 1)),
			end_column=int(loc.get("end_column", 0)),
		),
		evidence=str(item.get("evidence", "")),
		source=str(item.get("source", "tree-sitter")),
		symbol_summary=item.get("symbol_summary"),
		symbol_summary_embedding=item.get("symbol_summary_embedding"),
	)


def _dict_to_relationship(item: Mapping[str, Any] | Relationship) -> Relationship:
	if isinstance(item, Relationship):
		return item
	loc = item.get("location") or {}
	return Relationship(
		source=str(item["source"]),
		target=str(item["target"]),
		kind=str(item.get("kind", "related_to")),
		location=SourceLocation(
			path=str(loc.get("path", "")),
			line=int(loc.get("line", 1)),
			column=int(loc.get("column", 0)),
			end_line=int(loc.get("end_line", 1)),
			end_column=int(loc.get("end_column", 0)),
		),
		evidence=str(item.get("evidence", "")),
		source_index=str(item.get("source_index", "tree-sitter")),
		relationship_summary=item.get("relationship_summary"),
		relationship_summary_embedding=item.get("relationship_summary_embedding"),
	)


@log_call(logger)
def compare_graphs(
	existing_graph: Mapping[str, Any] | KnowledgeGraph,
	new_graph: Mapping[str, Any] | KnowledgeGraph,
) -> KnowledgeGraph:
	"""Return a delta KnowledgeGraph containing only new or modified files, symbols, and relationships."""
	old_dict = existing_graph.to_dict() if isinstance(existing_graph, KnowledgeGraph) else existing_graph
	new_dict = new_graph.to_dict() if isinstance(new_graph, KnowledgeGraph) else new_graph

	old_files = set(old_dict.get("files", []))
	old_symbols = {_symbol_key(s) for s in old_dict.get("symbols", [])}
	old_rels = {_rel_key(r) for r in old_dict.get("relationships", [])}

	new_symbols = new_graph.symbols if isinstance(new_graph, KnowledgeGraph) else [_dict_to_symbol(s) for s in new_dict.get("symbols", [])]
	new_rels = new_graph.relationships if isinstance(new_graph, KnowledgeGraph) else [_dict_to_relationship(r) for r in new_dict.get("relationships", [])]

	diff_symbols = [s for s in new_symbols if _symbol_key(s) not in old_symbols]
	diff_rels = [r for r in new_rels if _rel_key(r) not in old_rels]
	diff_files = sorted(
		(set(new_dict.get("files", [])) - old_files)
		| {s.location.path for s in diff_symbols}
		| {r.location.path for r in diff_rels}
	)

	return KnowledgeGraph(
		repository=str(new_dict.get("repository") or old_dict.get("repository") or ""),
		files=diff_files,
		symbols=diff_symbols,
		relationships=diff_rels,
		diagnostics=list(new_dict.get("diagnostics", [])),
	)


@log_call(logger)
def update_graph(
	graph: KnowledgeGraph,
	uri: str,
	username: str,
	password: str,
	database: str | None = None,
	export_path: str | Path | None = None,
) -> dict[str, int]:
	"""Export Neo4j to JSON, compare against `graph`, and persist only changed items."""
	existing = neo4j_to_json(uri, username, password, repository=graph.repository, database=database, output_path=export_path)
	delta = compare_graphs(existing, graph)
	if not delta.files and not delta.symbols and not delta.relationships:
		logger.info("No graph changes detected; skipping Neo4j write")
		return {"files": 0, "symbols": 0, "relationships": 0}
	return store_graph(delta, uri, username, password, database)