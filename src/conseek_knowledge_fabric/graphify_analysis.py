
from pathlib import Path
import json
from typing import Any

from graphify.detect import detect
from graphify.extract import collect_files, extract

from conseek_knowledge_fabric.models import KnowledgeGraph, Relationship, SourceLocation, Symbol


REPO_PATH = Path(
    ".conseek/repos/contextos-doc-translator"
).resolve()

OUTPUT_DIR = Path("graphify-out").resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "ast_graph_summarized.json"


def get_node_name(node):
    return (
        node.get("name")
        or node.get("label")
        or node.get("id")
        or "unknown symbol"
    )


def summarize_symbol(node):
    """Produce a deterministic summary from extracted metadata."""
    name = get_node_name(node)
    source_file = node.get("source_file", "unknown file")
    location = node.get("source_location", "unknown location")
    symbol_type = (
        node.get("type")
        or node.get("kind")
        or "code symbol"
    )

    docstring = node.get("docstring")
    if docstring:
        behavior = f" Documented behavior: {docstring}"
    else:
        behavior = (
            " Behavioral details are not established by "
            "the available extraction metadata."
        )

    return (
        f"{symbol_type.capitalize()} '{name}' is a code symbol "
        f"defined or referenced in {source_file} at {location}."
        f"{behavior}"
    )


def summarize_relationship(edge, nodes_by_id):
    """Explain the edge using its type and endpoint metadata."""
    source_id = edge.get("source")
    target_id = edge.get("target")

    source = nodes_by_id.get(source_id, {})
    target = nodes_by_id.get(target_id, {})

    source_name = get_node_name(source)
    target_name = get_node_name(target)

    relation = edge.get("relation", "related_to")
    provenance = edge.get("confidence", "UNKNOWN")

    explanations = {
        "calls": (
            f"'{source_name}' calls or references callable "
            f"'{target_name}' in the extracted call graph."
        ),
        "imports": (
            f"'{source_name}' imports or depends on "
            f"'{target_name}' according to the extracted graph."
        ),
        "uses": (
            f"'{source_name}' uses or references "
            f"'{target_name}'."
        ),
        "inherits": (
            f"'{source_name}' inherits from '{target_name}'."
        ),
        "contains": (
            f"'{source_name}' contains '{target_name}'."
        ),
        "defines": (
            f"'{source_name}' defines '{target_name}'."
        ),
    }

    explanation = explanations.get(
        relation,
        f"'{source_name}' has a '{relation}' relationship "
        f"with '{target_name}'."
    )

    if provenance == "AMBIGUOUS":
        explanation += (
            " The relationship is ambiguous and needs validation."
        )
    elif provenance == "INFERRED":
        explanation += (
            " This relationship is inferred, not a confirmed "
            "direct source-code fact."
        )

    return explanation


def enrich_ast_graph(graph_data):
    """Attach summary fields to graph nodes and edges so AST and graph exports are self-describing."""
    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])
    nodes_by_id = {node["id"]: node for node in nodes if node.get("id")}

    for node in nodes:
        node["symbol_summary"] = summarize_symbol(node)
        node["summary_provenance"] = "METADATA_TEMPLATE"

    for edge in edges:
        edge["relationship_summary"] = summarize_relationship(edge, nodes_by_id)
        edge["summary_provenance"] = "METADATA_TEMPLATE"

    return graph_data


def _location_from_metadata(path: str, location: Any) -> SourceLocation:
    if isinstance(location, dict):
        start = location.get("start", {})
        end = location.get("end", {})
        line = int(start.get("line", 1))
        column = int(start.get("column", 0))
        end_line = int(end.get("line", line))
        end_column = int(end.get("column", column))
    elif isinstance(location, (tuple, list)) and len(location) >= 4:
        line, column, end_line, end_column = [int(value) for value in location[:4]]
    else:
        line, column, end_line, end_column = 1, 0, 1, 0
    return SourceLocation(path=path, line=line, column=column, end_line=end_line, end_column=end_column)


def extract_repository(repository: str | Path, max_file_bytes: int = 1_000_000) -> KnowledgeGraph:
    """Compatibility wrapper that adapts the graphify extractor to the app's graph model."""
    root = Path(repository).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)

    detection = detect(root)
    code_files: list[Path] = []
    for item in detection.get("files", {}).get("code", []):
        path = Path(item)
        if not path.is_absolute():
            path = root / path
        if path.is_dir():
            code_files.extend(collect_files(path))
        elif path.is_file():
            code_files.append(path)

    code_files = sorted({path.resolve() for path in code_files if path.is_file()})
    if not code_files:
        raise RuntimeError(f"No code files detected under {root}")

    filtered_files = []
    for path in code_files:
        if path.stat().st_size <= max_file_bytes:
            filtered_files.append(path)

    graph_data = enrich_ast_graph(extract(filtered_files, root=root))
    graph = KnowledgeGraph(repository=str(root))

    for path in filtered_files:
        graph.files.append(path.relative_to(root).as_posix())

    for node in graph_data.get("nodes", []):
        node_id = str(node.get("id") or node.get("name") or "unknown")
        source_file = node.get("source_file") or node.get("file") or "unknown"
        target_path = source_file
        if isinstance(source_file, str):
            try:
                resolved = Path(source_file)
                if not resolved.is_absolute():
                    resolved = root / resolved
                target_path = resolved.relative_to(root).as_posix()
            except ValueError:
                target_path = source_file
        location = node.get("source_location") or node.get("location") or {}
        graph.symbols.append(
            Symbol(
                id=node_id,
                name=get_node_name(node),
                kind=node.get("type") or node.get("kind") or "symbol",
                language=(Path(target_path).suffix or ".txt").lstrip("."),
                location=_location_from_metadata(target_path, location),
                evidence=node.get("docstring") or node.get("summary") or json.dumps(node, ensure_ascii=False)[:500],
                symbol_summary=node.get("symbol_summary"),
            )
        )

    for edge in graph_data.get("edges", []):
        source_id = str(edge.get("source") or "unknown")
        target_id = str(edge.get("target") or "unknown")
        source_path = "unknown"
        if edge.get("source_file"):
            source_path = edge.get("source_file")
        relationship_summary = edge.get("relationship_summary") or edge.get("relation_summary")
        graph.relationships.append(
            Relationship(
                source=source_id,
                target=target_id,
                kind=edge.get("relation") or edge.get("kind") or "related_to",
                location=_location_from_metadata(source_path, edge.get("location") or {}),
                evidence=edge.get("evidence") or relationship_summary or "",
                relationship_summary=relationship_summary,
            )
        )

    for diag in graph_data.get("diagnostics", []):
        if diag:
            graph.diagnostics.append(str(diag))

    return graph


# def main():
#     if not REPO_PATH.is_dir():
#         raise FileNotFoundError(REPO_PATH)

#     # Step 1: Detect files.
#     detection = detect(REPO_PATH)

#     code_files = []
#     for item in detection.get("files", {}).get("code", []):
#         path = Path(item)

#         if not path.is_absolute():
#             path = REPO_PATH / path

#         if path.is_dir():
#             code_files.extend(collect_files(path))
#         elif path.is_file():
#             code_files.append(path)

#     # Step 2: Extract AST graph.
#     code_files = list(dict.fromkeys(p.resolve() for p in code_files))

#     if not code_files:
#         raise RuntimeError("No code files detected.")

#     graph_data = extract(code_files, root=REPO_PATH)

#     nodes = graph_data.get("nodes", [])
#     edges = graph_data.get("edges", [])

#     # Step 3: Build an ID index.
#     nodes_by_id = {
#         node["id"]: node
#         for node in nodes
#         if node.get("id")
#     }

#     # Step 4: Summarize every symbol.
#     for node in nodes:
#         node["symbol_summary"] = summarize_symbol(node)
#         node["summary_provenance"] = "METADATA_TEMPLATE"

#     # Step 5: Summarize every relationship.
#     for edge in edges:
#         edge["relationship_summary"] = summarize_relationship(
#             edge, nodes_by_id
#         )
#         edge["summary_provenance"] = "METADATA_TEMPLATE"

#     # Step 6: Validate edge endpoints.
#     missing_endpoints = [
#         edge
#         for edge in edges
#         if (
#             edge.get("source") not in nodes_by_id
#             or edge.get("target") not in nodes_by_id
#         )
#     ]

#     # Step 7: Export enriched extraction.
#     OUTPUT_FILE.write_text(
#         json.dumps(graph_data, indent=2, ensure_ascii=False),
#         encoding="utf-8",
#     )

#     print(f"Symbols summarized: {len(nodes)}")
#     print(f"Relationships summarized: {len(edges)}")
#     print(f"Edges with missing endpoints: {len(missing_endpoints)}")
#     print(f"Output: {OUTPUT_FILE}")


# if __name__ == "__main__":
#     main()