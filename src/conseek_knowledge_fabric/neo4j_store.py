"""Persist a Tree-sitter knowledge graph into Neo4j."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from neo4j import Driver
from neo4j.exceptions import Neo4jError

from conseek_knowledge_fabric.models import KnowledgeGraph
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
		SET edge.source_index = item.source_index,
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