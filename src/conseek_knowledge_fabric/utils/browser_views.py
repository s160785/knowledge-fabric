"""Build repository-scoped Neo4j Browser links for graph projections."""

from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from conseek_knowledge_fabric.utils.logging import get_logger, log_call


logger = get_logger(__name__)


VIEW_QUERIES = {
	"overview": "MATCH (r:Repository {path: $repo})-[:CONTAINS]->(f:File) OPTIONAL MATCH (f)-[d:DEFINES]->(s:Symbol) RETURN r,f,d,s LIMIT 500",
	"files": "MATCH (r:Repository {path: $repo})-[c:CONTAINS]->(f:File) RETURN r,c,f LIMIT 500",
	"symbols": "MATCH (r:Repository {path: $repo})-[:CONTAINS]->(f:File)-[d:DEFINES]->(s:Symbol) RETURN f,d,s LIMIT 500",
	"relationships": "MATCH (r:Repository {path: $repo})-[:CONTAINS]->(f:File)-[rel:CODE_RELATION]->(target) RETURN f,rel,target LIMIT 500",
}


@log_call(logger)
def _cypher_string(value: str) -> str:
	return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


@log_call(logger)
def browser_view_url(browser_url: str, repository: str, view: str, database: str | None = None) -> str:
	if view not in VIEW_QUERIES:
		logger.warning("Rejected unsupported graph view")
		raise ValueError(f"Unknown graph view: {view}")
	parsed = urlsplit(browser_url)
	path = parsed.path.rstrip("/")
	if not path.endswith("/browser"):
		path += "/browser"
	query = dict(parse_qsl(parsed.query, keep_blank_values=True))
	query["cmd"] = "play"
	query["arg"] = VIEW_QUERIES[view].replace("$repo", _cypher_string(repository))
	if database:
		query["db"] = database
	return urlunsplit((parsed.scheme, parsed.netloc, path + "/", urlencode(query), ""))