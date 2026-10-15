"""Command-line entry point for repository knowledge extraction."""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import dotenv_values

from conseek_knowledge_fabric.graphify_analysis import extract_repository
from conseek_knowledge_fabric.clone import resolve_repository
from conseek_knowledge_fabric.neo4j_store import store_graph
from conseek_knowledge_fabric.scip import enrich_from_scip
from conseek_knowledge_fabric.security import audit_repository
from conseek_knowledge_fabric.views.coding_agent import build as coding_agent_view
from conseek_knowledge_fabric.views.full import build as full_view


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract evidence-grounded code knowledge.")
    parser.add_argument("repository", help="Local repository path or remote Git URL")
    parser.add_argument("--clone-root", type=Path, default=Path(".conseek/repos"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"), help="Git credentials file for private repositories")
    parser.add_argument("--scip-index", type=Path, help="Optional SCIP index to add semantic references")
    parser.add_argument("--neo4j", action="store_true", help="Persist the extracted graph to Neo4j")
    parser.add_argument("--neo4j-database", help="Neo4j database name; defaults to the server default")
    parser.add_argument("--mode", choices=("full", "coding-agent"), default="full")
    parser.add_argument("--focus", action="append", default=[], help="Symbol name or ID to focus the coding-agent view on")
    parser.add_argument("--output", type=Path, help="Write JSON to this file instead of stdout")
    args = parser.parse_args()

    try:
        repository = resolve_repository(args.repository, args.clone_root, args.env_file)
        security_findings = audit_repository(repository)
        graph = extract_repository(repository)
        graph.diagnostics.extend(security_findings)
        if args.scip_index:
            enrich_from_scip(graph, args.scip_index)
        if args.neo4j:
            settings = {**dotenv_values(args.env_file), **os.environ}
            uri = settings.get("NEO4J_URI", "bolt://localhost:7687")
            username = settings.get("NEO4J_USERNAME", "neo4j")
            password = settings.get("NEO4J_PASSWORD")
            if not password:
                raise ValueError("Neo4j password is required; set NEO4J_PASSWORD in the environment or .env")
            counts = store_graph(graph, uri, username, password, args.neo4j_database or settings.get("NEO4J_DATABASE"))
            graph.diagnostics.append(
                f"Neo4j persisted {counts['files']} files, {counts['symbols']} symbols, "
                f"and {counts['relationships']} relationships"
            )
        response = full_view(graph) if args.mode == "full" else coding_agent_view(graph, set(args.focus))
        output = json.dumps(response, indent=2, ensure_ascii=False)
        if args.output:
            args.output.write_text(output + "\n", encoding="utf-8")
        else:
            print(output)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"conseek-extract: {error}", file=sys.stderr)
        raise SystemExit(2) from error