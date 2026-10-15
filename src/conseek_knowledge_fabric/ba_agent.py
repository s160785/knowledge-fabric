"""Repository BA agent backed by the Neo4j knowledge graph."""

from __future__ import annotations

import argparse
import os
from typing import Any

from neo4j import GraphDatabase


class RepoBAAgent:
    """Answer repo questions using evidence stored in Neo4j."""

    def __init__(
        self,
        uri: str | None = None,
        username: str | None = None,
        password: str | None = None,
        database: str | None = None,
    ) -> None:
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.username = username or os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = password if password is not None else os.getenv("NEO4J_PASSWORD")
        self.database = database or os.getenv("NEO4J_DATABASE")
        self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password or ""))

    def close(self) -> None:
        self.driver.close()

    def _safe_rows(self, result: Any) -> list[dict[str, Any]]:
        try:
            rows = result.data()
        except AttributeError:
            rows = list(result)
        if not isinstance(rows, list):
            return []
        return [dict(row) for row in rows]

    def _repo_question_query(self, question: str) -> tuple[str, dict[str, Any]]:
        text = question.lower()

        if any(token in text for token in ["entry point", "entrypoint", "main function", "startup", "main"]):
            return (
                """
                MATCH (repo:Repository)-[:CONTAINS]->(file:File)-[:DEFINES]->(symbol:Symbol)
                WHERE toLower(symbol.name) IN ['main', 'app', 'run', 'cli']
                RETURN repo.path AS repository, collect(DISTINCT symbol.name) AS names, collect(DISTINCT file.path) AS files
                ORDER BY repository
                LIMIT 20
                """,
                {},
            )

        if any(token in text for token in ["dependency", "depends on", "import", "imports", "edges", "graph"]):
            return (
                """
                MATCH (repo:Repository)-[:CONTAINS]->(:File)-[rel:CODE_RELATION]->(target)
                WHERE rel.kind = 'imports' OR rel.kind = 'calls'
                RETURN repo.path AS repository, rel.kind AS kind, count(rel) AS count
                ORDER BY count DESC
                LIMIT 20
                """,
                {},
            )

        if any(token in text for token in ["files", "modules", "components", "architecture"]):
            return (
                """
                MATCH (repo:Repository)-[:CONTAINS]->(file:File)
                RETURN repo.path AS repository, count(DISTINCT file) AS file_count, collect(DISTINCT file.path)[..10] AS sample_files
                ORDER BY file_count DESC
                LIMIT 20
                """,
                {},
            )

        if any(token in text for token in ["call", "calls", "who calls", "caller", "callee"]):
            return (
                """
                MATCH (repo:Repository)-[:CONTAINS]->(:File)-[:DEFINES]->(source:Symbol)-[rel:CODE_RELATION]->(target)
                WHERE rel.kind = 'calls'
                RETURN repo.path AS repository, source.name AS source_name, target.name AS target_name, count(rel) AS call_count
                ORDER BY call_count DESC
                LIMIT 20
                """,
                {},
            )

        return (
            """
            MATCH (repo:Repository)-[:CONTAINS]->(file:File)
            OPTIONAL MATCH (file)-[:DEFINES]->(symbol:Symbol)
            RETURN repo.path AS repository,
                   count(DISTINCT file) AS file_count,
                   count(DISTINCT symbol) AS symbol_count,
                   collect(DISTINCT file.path)[..10] AS sample_files
            ORDER BY file_count DESC
            LIMIT 10
            """,
            {},
        )

    def _summarize_rows(self, question: str, rows: list[dict[str, Any]]) -> str:
        if not rows:
            return "I could not find evidence for that question in the Neo4j graph."

        text = question.lower()
        if any(token in text for token in ["entry point", "entrypoint", "main function", "startup", "main"]):
            names: set[str] = set()
            files: set[str] = set()
            for row in rows:
                for name in row.get("names", []) or []:
                    names.add(str(name))
                for file in row.get("files", []) or []:
                    files.add(str(file))
            if not names:
                return "The graph does not show obvious entry-point symbols for the repositories in Neo4j."
            return (
                f"The graph shows likely entry points: {', '.join(sorted(names)[:10])}. "
                f"They appear in {', '.join(sorted(files)[:10])}."
            )

        if any(token in text for token in ["dependency", "depends on", "import", "imports", "graph"]):
            summaries: list[str] = []
            for row in rows[:5]:
                summaries.append(f"{row.get('repository')} has {row.get('count')} {row.get('kind')} relationships")
            return "; ".join(summaries) if summaries else "The graph does not show dependency information."

        if any(token in text for token in ["files", "modules", "components", "architecture"]):
            summaries = []
            for row in rows[:5]:
                repo = row.get("repository") or "unknown"
                file_count = row.get("file_count", 0)
                files = row.get("sample_files") or []
                summaries.append(f"{repo} has {file_count} files, including {', '.join(map(str, files[:3]))}")
            return "; ".join(summaries) if summaries else "The graph does not contain file-level architecture data."

        if any(token in text for token in ["call", "calls", "caller", "callee"]):
            summaries = []
            for row in rows[:5]:
                summaries.append(f"{row.get('source_name')} calls {row.get('target_name')} {row.get('call_count')} times")
            return "; ".join(summaries) if summaries else "The graph does not show call relationships."

        summaries = []
        for row in rows[:5]:
            repo = row.get("repository") or "unknown"
            file_count = row.get("file_count", 0)
            symbol_count = row.get("symbol_count", 0)
            summaries.append(f"{repo} contains {file_count} files and {symbol_count} symbols")
        return "; ".join(summaries) if summaries else "The graph does not contain repo metadata for that question."

    def ask(self, question: str) -> dict[str, Any]:
        query, params = self._repo_question_query(question)
        with self.driver.session(database=self.database) as session:
            result = session.run(query, **params)
            rows = self._safe_rows(result)
        answer = self._summarize_rows(question, rows)
        return {
            "question": question,
            "answer": answer,
            "evidence": rows[:10],
            "query": query,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a repository question using the Neo4j code graph.")
    parser.add_argument("question", help="Natural-language question to answer")
    parser.add_argument("--uri", default=None, help="Neo4j URI; default from NEO4J_URI or bolt://localhost:7687")
    parser.add_argument("--username", default=None, help="Neo4j username; default from NEO4J_USERNAME or neo4j")
    parser.add_argument("--password", default=None, help="Neo4j password; default from NEO4J_PASSWORD")
    parser.add_argument("--database", default=None, help="Neo4j database; default from NEO4J_DATABASE")
    args = parser.parse_args()

    agent = RepoBAAgent(args.uri, args.username, args.password, args.database)
    try:
        response = agent.ask(args.question)
        print(response["answer"])
    finally:
        agent.close()


if __name__ == "__main__":
    main()
