"""Orchestrate repository scanning, Tree-sitter extraction, and Neo4j writes."""

import os
from pathlib import Path
from typing import Any

from dotenv import dotenv_values

from conseek_knowledge_fabric.graphify_analysis import extract_repository, update_knowledge_graph
from conseek_knowledge_fabric.clone import resolve_repository
from conseek_knowledge_fabric.neo4j_store import store_graph, update_graph
from conseek_knowledge_fabric.security import audit_repository
from conseek_knowledge_fabric.utils.logging import get_logger, log_call


logger = get_logger(__name__)


@log_call(logger)
def build_repository_graph(
	repository_source: str,
	neo4j_uri: str,
	neo4j_username: str,
	neo4j_password: str,
	neo4j_database: str | None = None,
	branch: str | None = None,
) -> dict[str, Any]:
	logger.info("Resolving repository and selected branch")
	env_file = Path(os.getenv("CODEFABRIC_ENV_FILE", ".env"))
	repository = resolve_repository(repository_source, ".codefabric/repos", env_file, branch)
	logger.info("Running repository audit")
	diagnostics = audit_repository(repository)
	logger.info("Updating knowledge graph with diff against Neo4j")
	result = update_knowledge_graph(
		repository, neo4j_uri, neo4j_username, neo4j_password, neo4j_database
	)
	counts = result["counts"]
	logger.info(
		"Persistence complete files=%d symbols=%d relationships=%d",
		counts["files"], counts["symbols"], counts["relationships"],
	)
	return {"repository": result["repository"], "counts": counts, "diagnostics": diagnostics + result.get("diagnostics", [])}