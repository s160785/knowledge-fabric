from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any

import requests
from neo4j import GraphDatabase, Query
from neo4j.warnings import Neo4jWarning

from conseek_knowledge_fabric.utils.logging import get_logger, log_call
from conseek_knowledge_fabric.utils.redis import RedisMemory

import warnings

# Ignore all warnings including Neo4j DBMS notification warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=Neo4jWarning)

logger = get_logger(__name__)


class RepoBAChatbot:
    """Repository BA Chatbot utilizing Neo4j embeddings, Graph RAG, and Redis cache memory."""

    def __init__(
        self,
        uri: str | None = None,
        username: str | None = None,
        password: str | None = None,
        database: str | None = None,
        redis_url: str | None = None,
        ollama_url: str | None = None,
        model: str | None = None,
        embedding_model: str | None = None,
    ) -> None:
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.username = username or os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = password if password is not None else os.getenv("NEO4J_PASSWORD", "")
        self.database = database or os.getenv("NEO4J_DATABASE") or None

        self.driver = GraphDatabase.driver(
            self.uri,
            auth=(self.username, self.password),
            notifications_min_severity="OFF",
        )

        self.ollama_url = (ollama_url or os.getenv("OLLAMA_URL", "http://localhost:11434")).rstrip("/")
        self.model = model or os.getenv("CONTEXT_GEN_OLLAMA_MODEL", "gemma3:4b")
        self.embedding_model = embedding_model or os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text:latest")
        self.max_history = 12

        logger.info(
            "Initializing RepoBAChatbot (neo4j_uri=%s, database=%s, model=%s, embedding_model=%s)",
            self.uri,
            self.database or "default",
            self.model,
            self.embedding_model,
        )

        self.memory = RedisMemory(redis_url=redis_url)
        self._ensure_vector_index()

    @property
    def history(self) -> list[dict[str, str]]:
        return self.memory.get_history("default", limit=self.max_history)

    @history.setter
    def history(self, value: list[dict[str, str]]) -> None:
        self.memory.clear_history("default")
        for item in value:
            self.memory.add_history("default", item.get("role", "user"), item.get("content", ""))

    def close(self) -> None:
        logger.info("Closing Neo4j driver connection")
        self.driver.close()

    def _ensure_vector_index(self) -> None:
        """Create vector index in Neo4j if supported and not exists."""
        try:
            logger.info("Checking/creating vector index 'symbol_embeddings' in Neo4j")
            with self.driver.session(database=self.database) as session:
                session.run(
                    """
                    CREATE VECTOR INDEX symbol_embeddings IF NOT EXISTS
                    FOR (s:Symbol) ON (s.symbol_summary_embedding)
                    OPTIONS {indexConfig: {`vector.dimensions`: 768, `vector.similarity_function`: 'cosine'}}
                    """
                ).consume()
            logger.info("Vector index 'symbol_embeddings' verified/created")
        except Exception as exc:
            logger.debug("Vector index check skipped or unsupported: %s", exc)

    def _safe_rows(self, result: Any) -> list[dict[str, Any]]:
        """Extract rows from Neo4j result, data() method, or iterable."""
        try:
            data = result.data()
            if isinstance(data, list):
                return [dict(r) for r in data]
        except AttributeError:
            pass
        try:
            return [dict(record) for record in result]
        except Exception:
            return []

    # ---------------------------------------------------------
    # Embeddings & LLM
    # ---------------------------------------------------------

    @log_call(logger)
    def _get_embedding(self, text: str) -> list[float] | None:
        """Generate text embedding with Ollama, using Redis cache if available."""
        logger.info("Generating embedding for text: %r", text[:60] + ("..." if len(text) > 60 else ""))
        cached = self.memory.get_embedding(text)
        if cached:
            logger.info("Embedding cache HIT in Redis/memory (dimensions=%d)", len(cached))
            return cached

        logger.info("Embedding cache MISS; calling Ollama model=%s", self.embedding_model)
        for endpoint, payload in [
            ("/api/embeddings", {"model": self.embedding_model, "prompt": text}),
            ("/api/embed", {"model": self.embedding_model, "input": text}),
        ]:
            try:
                resp = requests.post(f"{self.ollama_url}{endpoint}", json=payload, timeout=30)
                if resp.ok:
                    data = resp.json()
                    emb = data.get("embedding")
                    if not emb and "embeddings" in data and data["embeddings"]:
                        emb = data["embeddings"][0]
                    if emb and isinstance(emb, list):
                        logger.info("Generated embedding successfully from endpoint %s (dimensions=%d)", endpoint, len(emb))
                        self.memory.set_embedding(text, emb)
                        return emb
            except Exception as exc:
                logger.debug("Ollama embedding endpoint %s failed: %s", endpoint, exc)
                continue

        logger.warning("Failed to generate embedding from Ollama for text: %r", text[:60])
        return None

    @log_call(logger)
    def _llm(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
    ) -> str:
        logger.info(
            "Calling Ollama chat API (url=%s, model=%s, messages=%d, temperature=%.2f)",
            self.ollama_url,
            self.model,
            len(messages),
            temperature,
        )
        response = requests.post(
            f"{self.ollama_url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature},
            },
            timeout=120,
        )
        if not response.ok:
            try:
                err_detail = response.json().get("error")
            except Exception:
                err_detail = response.text
            logger.error("Ollama chat API returned error status=%d: %s", response.status_code, err_detail)
            raise RuntimeError(f"Ollama error ({response.status_code}): {err_detail}")

        content = response.json()["message"]["content"].strip()
        logger.info("Received Ollama chat response (length=%d characters)", len(content))
        return content

    # ---------------------------------------------------------
    # Conversation understanding & intent
    # ---------------------------------------------------------

    def _is_general_conversation(self, text: str) -> bool:
        """Identify greetings, pleasantries, small talk, and assistant inquiries."""
        clean = text.lower().strip()
        # Direct small talk matches
        common_chit_chat = {
            "hi", "hello", "hey", "hiya", "howdy", "good morning", "good afternoon",
            "good evening", "how are you", "how are you doing", "what's up", "whats up",
            "thanks", "thank you", "thx", "appreciate it", "great", "awesome", "cool",
            "ok", "okay", "bye", "goodbye", "see you", "who are you", "what are you",
            "what can you do", "help", "help me", "what is your role", "what do you do"
        }
        if clean in common_chit_chat:
            return True

        # Matches phrases like "hello there", "thank you so much", etc.
        patterns = [
            r"^(hi|hello|hey)\b",
            r"^(thanks|thank you)\b",
            r"^(who|what) are you\b",
            r"^what can you (do|help)\b",
            r"^how are you\b",
            r"^(good morning|good afternoon|good evening)\b",
        ]
        return any(re.search(pat, clean) for pat in patterns)

    @log_call(logger)
    def _resolve_question(self, question: str, session_id: str = "default") -> str:
        """
        Resolve references such as 'it', 'that method', and
        'what about its dependencies' using conversation history.
        """
        if self._is_general_conversation(question):
            logger.info("General conversation detected; skipping question resolution")
            return question

        history = self.memory.get_history(session_id, limit=self.max_history)
        if not history:
            logger.info("No prior conversation history for session_id=%s; skipping rewrite", session_id)
            return question

        # If question has no pronouns / referential words, keep it as-is
        pronouns = {"it", "its", "that", "this", "they", "them", "these", "those", "previous", "earlier", "above"}
        words = set(re.findall(r"\b[a-z]+\b", question.lower()))
        if not (words & pronouns) and len(words) >= 3:
            logger.info("Question is already standalone (no pronouns); skipping rewrite")
            return question

        logger.info(
            "Resolving conversational references using %d history turns for session_id=%s",
            len(history),
            session_id,
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You rewrite the user's latest question to resolve references like 'it', 'that', 'they', or 'its' "
                    "using previous conversation turns. "
                    "Return ONLY the rewritten, standalone question in one short sentence. "
                    "Do NOT answer the question. Do NOT include previous assistant summaries or explanations."
                ),
            },
            *history[-4:],
            {
                "role": "user",
                "content": question,
            },
        ]
        try:
            resolved = self._llm(messages, temperature=0.0)
            if resolved and len(resolved.splitlines()) == 1 and len(resolved) < 200:
                logger.info("Rewrote question from %r to %r", question, resolved)
                return resolved
            logger.info("LLM rewrite output unusual; retaining original question %r", question)
            return question
        except Exception as exc:
            logger.warning("Conversation resolution rewrite failed (%s); using original question", exc)
            return question

    # ---------------------------------------------------------
    # Neo4j Graph RAG retrieval
    # ---------------------------------------------------------

    @log_call(logger)
    def _retrieve_evidence(
        self, question: str
    ) -> list[dict[str, Any]]:
        """
        Neo4j Graph RAG retrieval:
        1. Vector search using symbol embeddings in Neo4j (via index or cosine similarity).
        2. Graph traversal expanding adjacent relationships and connected symbols.
        3. Keyword matching fallback / hybrid merge.
        """
        logger.info("Starting Neo4j evidence retrieval for query: %r", question)
        q_lower = question.lower()
        words = list(dict.fromkeys(
            re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{2,}", q_lower)
        ))
        stopwords = {
            "what", "which", "where", "when", "does", "this",
            "that", "with", "from", "have", "about", "explain",
            "please", "show", "tell", "could", "would", "should",
            "repository", "function", "method", "class", "code",
        }
        terms = [word for word in words if word not in stopwords][:15]
        logger.info("Extracted %d search terms for graph querying: %s", len(terms), terms)

        evidence_by_id: dict[str, dict[str, Any]] = {}

        # Check for scope/architecture intent if asked
        if any(token in q_lower for token in ["scope", "architecture", "overview", "what is this", "what does this do", "explain this project"]):
            logger.info("Query matched scope/overview intent; querying repository modules and files")
            scope_query = """
            MATCH (repo:Repository)-[:CONTAINS]->(file:File)
            OPTIONAL MATCH (file)-[:DEFINES]->(s:Symbol)
            WHERE NOT file.path CONTAINS 'test' AND NOT s.name STARTS WITH 'test_'
            WITH repo, file, count(s) as symbol_count, collect(s.name)[..5] as top_symbols
            RETURN
                repo.path AS repository,
                file.path AS file,
                'Module' AS kind,
                file.path AS symbol,
                file.path AS name,
                file.name AS symbol_id,
                'File' AS symbol_labels,
                '' AS signature,
                'Contains ' + toString(symbol_count) + ' symbols including: ' + apoc.text.join(top_symbols, ', ') AS symbol_summary,
                [] AS relationships,
                1.0 AS score
            ORDER BY symbol_count DESC
            LIMIT 10
            """
            try:
                with self.driver.session(database=self.database) as session:
                    rows = self._safe_rows(session.run(scope_query))
                    logger.info("Found %d high-level scope/module candidates in Neo4j", len(rows))
                    for row in rows:
                        sid = row.get("symbol_id") or row.get("file")
                        if sid:
                            evidence_by_id[str(sid)] = row
            except Exception as exc:
                logger.debug("Scope/Overview query failed: %s", exc)

        # Check for entry points if asked
        elif any(token in q_lower for token in ["entry point", "entrypoint", "main function", "startup", "main"]):
            logger.info("Query matched entry-point intent; querying designated entry-point symbols")
            entry_query = """
            MATCH (repo:Repository)-[:CONTAINS]->(file:File)-[:DEFINES]->(s:Symbol)
            WHERE toLower(s.name) IN ['main', 'app', 'run', 'cli', 'index', 'server', 'start']
              AND NOT file.path CONTAINS 'test'
            OPTIONAL MATCH (s)-[r:CODE_RELATION]-(other)
            RETURN
                repo.path AS repository,
                file.path AS file,
                elementId(s) AS symbol_id,
                coalesce(properties(s)['name'], '') AS symbol,
                coalesce(properties(s)['name'], '') AS name,
                labels(s) AS symbol_labels,
                coalesce(properties(s)['kind'], 'Function') AS kind,
                coalesce(properties(s)['signature'], '') AS signature,
                coalesce(properties(s)['symbol_summary'], properties(s)['summary'], properties(s)['description'], '') AS symbol_summary,
                collect(DISTINCT CASE WHEN r IS NULL THEN null ELSE {
                    direction: CASE WHEN startNode(r) = s THEN 'outgoing' ELSE 'incoming' END,
                    kind: coalesce(properties(r)['kind'], type(r)),
                    summary: coalesce(properties(r)['summary'], properties(r)['relationship_summary'], ''),
                    target: coalesce(properties(other)['name'], properties(other)['path'], '')
                } END)[..10] AS relationships,
                1.0 AS score
            LIMIT 10
            """
            try:
                with self.driver.session(database=self.database) as session:
                    rows = self._safe_rows(session.run(entry_query))
                    logger.info("Found %d entry-point symbol candidates in Neo4j", len(rows))
                    for row in rows:
                        sid = row.get("symbol_id") or row.get("symbol") or row.get("name")
                        if sid:
                            evidence_by_id[str(sid)] = row
            except Exception as exc:
                logger.debug("Entry point query failed: %s", exc)

        # Vector RAG search with embeddings in Neo4j
        logger.info("Generating query embedding for Neo4j Vector RAG retrieval")
        embedding = self._get_embedding(question)
        if embedding:
            logger.info("Executing vector index similarity search in Neo4j (limit=8)")
            # Attempt 1: Neo4j Vector Index
            vector_index_query = """
            CALL db.index.vector.queryNodes('symbol_embeddings', $limit, $embedding)
            YIELD node AS s, score
            MATCH (repo:Repository)-[:CONTAINS]->(file:File)-[:DEFINES]->(s)
            WHERE NOT file.path CONTAINS 'test' AND NOT s.name STARTS WITH 'test_'
            OPTIONAL MATCH (s)-[r:CODE_RELATION]-(other)
            RETURN
                repo.path AS repository,
                file.path AS file,
                elementId(s) AS symbol_id,
                coalesce(properties(s)['name'], '') AS symbol,
                coalesce(properties(s)['name'], '') AS name,
                labels(s) AS symbol_labels,
                coalesce(properties(s)['kind'], '') AS kind,
                coalesce(properties(s)['signature'], '') AS signature,
                coalesce(properties(s)['symbol_summary'], properties(s)['summary'], properties(s)['description'], '') AS symbol_summary,
                collect(DISTINCT CASE WHEN r IS NULL THEN null ELSE {
                    direction: CASE WHEN startNode(r) = s THEN 'outgoing' ELSE 'incoming' END,
                    kind: coalesce(properties(r)['kind'], type(r)),
                    summary: coalesce(properties(r)['summary'], properties(r)['relationship_summary'], ''),
                    target: coalesce(properties(other)['name'], properties(other)['path'], ''),
                    target_summary: coalesce(properties(other)['summary'], properties(other)['symbol_summary'], properties(other)['description'], '')
                } END)[..15] AS relationships,
                score
            ORDER BY score DESC
            """
            try:
                with self.driver.session(database=self.database) as session:
                    rows = self._safe_rows(session.run(vector_index_query, limit=8, embedding=embedding))
                    logger.info("Vector index query returned %d matches", len(rows))
                    for row in rows:
                        sid = row.get("symbol_id") or row.get("symbol") or row.get("name")
                        if sid:
                            evidence_by_id[str(sid)] = row
            except Exception as exc:
                logger.info("Vector index search unavailable (%s); trying cosine similarity function fallback", exc)
                # Attempt 2: Cypher vector similarity function fallback
                vector_func_query = """
                MATCH (repo:Repository)-[:CONTAINS]->(file:File)-[:DEFINES]->(s:Symbol)
                WHERE properties(s)['symbol_summary_embedding'] IS NOT NULL
                  AND NOT file.path CONTAINS 'test' AND NOT s.name STARTS WITH 'test_'
                WITH repo, file, s,
                     vector.similarity.cosine(properties(s)['symbol_summary_embedding'], $embedding) AS score
                WHERE score > 0.2
                ORDER BY score DESC
                LIMIT $limit
                OPTIONAL MATCH (s)-[r:CODE_RELATION]-(other)
                RETURN
                    repo.path AS repository,
                    file.path AS file,
                    elementId(s) AS symbol_id,
                    coalesce(properties(s)['name'], '') AS symbol,
                    coalesce(properties(s)['name'], '') AS name,
                    labels(s) AS symbol_labels,
                    coalesce(properties(s)['kind'], '') AS kind,
                    coalesce(properties(s)['signature'], '') AS signature,
                    coalesce(properties(s)['symbol_summary'], properties(s)['summary'], properties(s)['description'], '') AS symbol_summary,
                    collect(DISTINCT CASE WHEN r IS NULL THEN null ELSE {
                        direction: CASE WHEN startNode(r) = s THEN 'outgoing' ELSE 'incoming' END,
                        kind: coalesce(properties(r)['kind'], type(r)),
                        summary: coalesce(properties(r)['summary'], properties(r)['relationship_summary'], ''),
                        target: coalesce(properties(other)['name'], properties(other)['path'], ''),
                        target_summary: coalesce(properties(other)['summary'], properties(other)['symbol_summary'], properties(other)['description'], '')
                    } END)[..15] AS relationships,
                    score
                ORDER BY score DESC
                """
                try:
                    with self.driver.session(database=self.database) as session:
                        rows = self._safe_rows(session.run(vector_func_query, limit=8, embedding=embedding))
                        logger.info("Cosine similarity query returned %d matches", len(rows))
                        for row in rows:
                            sid = row.get("symbol_id") or row.get("symbol") or row.get("name")
                            if sid:
                                evidence_by_id[str(sid)] = row
                except Exception as func_exc:
                    logger.debug("Cosine similarity query failed: %s", func_exc)
        else:
            logger.info("Skipping vector search as embedding is unavailable")

        # Keyword / Hybrid Graph search
        if terms or not evidence_by_id:
            logger.info("Executing keyword graph query in Neo4j with terms=%s", terms)
            keyword_query = """
            MATCH (repo:Repository)-[:CONTAINS]->(file:File)
                  -[:DEFINES]->(s:Symbol)
            WHERE NOT file.path CONTAINS 'test' AND NOT s.name STARTS WITH 'test_'

            WITH repo, file, s,
                 coalesce(properties(s)['name'], '') AS symbol_name,
                 coalesce(properties(s)['summary'], properties(s)['symbol_summary'],
                          properties(s)['description'], '') AS symbol_summary,
                 coalesce(properties(s)['signature'], '') AS signature

            WITH repo, file, s, symbol_name, symbol_summary, signature,
                 size([term IN $terms
                       WHERE toLower(symbol_name) CONTAINS term
                          OR toLower(symbol_summary) CONTAINS term
                          OR toLower(signature) CONTAINS term]) AS score

            WHERE score > 0 OR size($terms) = 0

            WITH repo, file, s, symbol_name, symbol_summary,
                 signature, score
            ORDER BY score DESC, symbol_name
            LIMIT 8

            OPTIONAL MATCH (s)-[r:CODE_RELATION]-(other)

            RETURN
                repo.path AS repository,
                file.path AS file,
                elementId(s) AS symbol_id,
                symbol_name AS symbol,
                symbol_name AS name,
                labels(s) AS symbol_labels,
                coalesce(s.kind, '') AS kind,
                signature AS signature,
                symbol_summary AS symbol_summary,
                collect(DISTINCT CASE WHEN r IS NULL THEN null ELSE {
                    direction: CASE
                        WHEN startNode(r) = s THEN 'outgoing'
                        ELSE 'incoming'
                    END,
                    kind: coalesce(properties(r)['kind'], type(r)),
                    summary: coalesce(
                        properties(r)['summary'], properties(r)['relationship_summary'], ''
                    ),
                    target: coalesce(properties(other)['name'], properties(other)['path'], ''),
                    target_summary: coalesce(
                        properties(other)['summary'],
                        properties(other)['symbol_summary'],
                        properties(other)['description'],
                        ''
                    )
                } END)[..15] AS relationships,
                score
            ORDER BY score DESC
            """
            try:
                with self.driver.session(database=self.database) as session:
                    rows = self._safe_rows(session.run(keyword_query, terms=terms))
                    logger.info("Keyword graph search returned %d candidate symbols", len(rows))
                    for row in rows:
                        sid = row.get("symbol_id") or row.get("symbol") or row.get("name")
                        if sid and str(sid) not in evidence_by_id:
                            evidence_by_id[str(sid)] = row
            except Exception as exc:
                logger.debug("Keyword graph query failed: %s", exc)

        results = sorted(evidence_by_id.values(), key=lambda r: r.get("score", 0), reverse=True)
        logger.info("Retrieved %d unique evidence items from Neo4j (top %d retained)", len(evidence_by_id), min(len(results), 10))
        return results[:10]

    @log_call(logger)
    def _retrieve_repository_overview(self) -> list[dict[str, Any]]:
        logger.info("Querying repository overview fallback from Neo4j")
        query = """
        MATCH (repo:Repository)-[:CONTAINS]->(file:File)
        OPTIONAL MATCH (file)-[:DEFINES]->(symbol:Symbol)
        RETURN
            repo.path AS repository,
            count(DISTINCT file) AS file_count,
            count(DISTINCT symbol) AS symbol_count,
            collect(DISTINCT file.path)[..10] AS sample_files
        ORDER BY file_count DESC
        LIMIT 10
        """
        try:
            with self.driver.session(database=self.database) as session:
                rows = self._safe_rows(session.run(query))
                logger.info("Retrieved repository overview with %d entries", len(rows))
                return rows
        except Exception as exc:
            logger.warning("Repository overview query failed: %s", exc)
            return []

    # ---------------------------------------------------------
    # Answer generation
    # ---------------------------------------------------------

    @log_call(logger)
    def _summarize_evidence(self, question: str, evidence: list[dict[str, Any]]) -> str:
        """Fallback natural-language summary when LLM is unavailable."""
        logger.info("Generating fallback evidence summary for %d items", len(evidence))
        q = question.lower().strip()
        if q in {"hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening"}:
            return (
                "Hello! I am your Business Analyst assistant for this software project. "
                "I can explain how the application works, what features exist, and how information flows "
                "through the system in simple, everyday language without technical jargon. "
                "How can I help you today?"
            )

        if not evidence:
            return (
                "I couldn't find specific information about that in the application knowledge base. "
                "Feel free to ask about the main features, entry points, or general workflows in the project!"
            )

        if any(token in q for token in ["entry point", "entrypoint", "main function", "startup", "main"]):
            names = [row.get("symbol") or row.get("name") for row in evidence if row.get("symbol") or row.get("name")]
            names_str = ", ".join(sorted(set(str(n) for n in names if n))[:8]) or "core services"
            return (
                f"The main entry points where the application starts and initiates its workflows include: {names_str}. "
                "These components kick off the system's primary user interactions and background services."
            )

        capabilities = []
        for row in evidence[:5]:
            sym = row.get("symbol") or row.get("name")
            summary = row.get("symbol_summary") or row.get("summary")
            if summary and len(summary) > 15:
                capabilities.append(f"{sym}: {summary}")
            elif sym:
                capabilities.append(f"{sym}")

        if capabilities:
            bullet_points = "\n".join(f"• {c}" for c in capabilities)
            return (
                f"Here are the key capabilities and components found in the system:\n{bullet_points}\n\n"
                "Let me know if you would like me to explain any of these features in more detail!"
            )

        return "The system contains several interconnected components. Would you like to explore a specific feature or workflow?"

    @log_call(logger)
    def _answer(
        self,
        original_question: str,
        resolved_question: str,
        evidence: list[dict[str, Any]],
        session_id: str = "default",
    ) -> str:
        logger.info(
            "Generating answer (original_question=%r, resolved_question=%r, evidence_items=%d, session_id=%s)",
            original_question,
            resolved_question,
            len(evidence),
            session_id,
        )
        history = self.memory.get_history(session_id, limit=self.max_history)
        is_general = self._is_general_conversation(original_question)

        if evidence:
            logger.info("Formatting %d Neo4j evidence items as prompt context", len(evidence))
            context = json.dumps(evidence, indent=2, default=str)
        elif is_general:
            logger.info("General conversation query; omitting repository context")
            context = "General conversational query. No repository lookup needed."
        else:
            logger.info("No specific evidence items found; using repository overview as prompt context")
            context = json.dumps(
                self._retrieve_repository_overview(),
                indent=2,
                default=str,
            )

        messages = [
            {
                "role": "system",
                "content": """
You are an approachable, friendly Business Analyst (BA) and product specialist for this software application.
You are conversing with non-technical people (business stakeholders, product managers, end-users, or clients) who do NOT know how to code and do NOT know what is written in the code.

Persona & Communication Guidelines:
1. Tone: Conversational, clear, friendly, and non-technical. Speak like a helpful product teammate.
2. No Code Jargon: Avoid technical code terms like 'AST', 'elementId', 'methods', 'classes', 'parameters', 'decorators', 'signatures', or raw file paths, unless specifically asked.
3. Translate Code into Business Value:
   - Describe WHAT the system does, WHO it helps, and WHAT business workflows it performs.
   - For example, instead of saying "Symbol `auth.login()` has a call relationship to `jwt.verify()`", say "The application includes a secure sign-in process that verifies user identity before granting access."
   - If asked about "entry points", explain them in business terms as the primary starting doors or launchpads of the application (e.g. web portal, background processing service, or command line).
4. Casual Chat & Greetings: If the user says 'hi', 'hello', 'who are you', or makes small talk, respond warmly and naturally as their BA guide, explaining how you can help them understand the application.
5. Factual Grounding: Base all functional claims on the provided Neo4j evidence. Do not invent imaginary business features.
6. Formatting: Use clean, readable paragraphs or bullet points. Avoid raw JSON, database query syntax, or code blocks.
""".strip(),
            },
            *history,
            {
                "role": "user",
                "content": (
                    f"User question: {original_question}\n"
                    f"Underlying context/question: {resolved_question}\n\n"
                    f"Knowledge base evidence from repository:\n{context}\n\n"
                    "Respond to the user in a friendly, conversational manner suitable for a non-technical person."
                ),
            },
        ]

        try:
            answer = self._llm(messages)
            logger.info("LLM successfully generated answer (%d characters)", len(answer))
            return answer
        except Exception as exc:
            logger.warning("LLM response generation failed (%s); returning grounded evidence summary", exc)
            return self._summarize_evidence(original_question, evidence)

    # ---------------------------------------------------------
    # Public chatbot interface
    # ---------------------------------------------------------

    @log_call(logger)
    def ask(self, question: str, session_id: str = "default") -> dict[str, Any]:
        question = question.strip()
        logger.info("Chatbot received question [session_id=%s]: %r", session_id, question)
        if not question:
            logger.info("Empty question received; returning early prompt")
            return {"question": "", "resolved_question": "", "answer": "Please enter a question.", "evidence": []}

        # Check QA cache in Redis memory
        cached = self.memory.get_qa_cache(f"{session_id}:{question}")
        if cached:
            logger.info("QA cache HIT in Redis/memory for question [session_id=%s]; returning cached result", session_id)
            return {**cached, "cached": True}

        logger.info("QA cache MISS for question [session_id=%s]", session_id)
        is_general = self._is_general_conversation(question)

        logger.info("[Step 1/4] Resolving question conversational context")
        resolved = self._resolve_question(question, session_id=session_id)

        if is_general:
            logger.info("Skipping Neo4j retrieval for general conversational query")
            evidence: list[dict[str, Any]] = []
        else:
            logger.info("[Step 2/4] Retrieving graph evidence from Neo4j")
            evidence = self._retrieve_evidence(resolved)

        logger.info("[Step 3/4] Generating answer from retrieved evidence")
        answer = self._answer(question, resolved, evidence, session_id=session_id)

        logger.info("[Step 4/4] Updating conversation history in Redis memory")
        self.memory.add_history(session_id, "user", question)
        self.memory.add_history(session_id, "assistant", answer)

        result = {
            "question": question,
            "resolved_question": resolved,
            "answer": answer,
            "evidence": evidence,
        }

        logger.info("Caching answer in Redis memory [session_id=%s]", session_id)
        self.memory.set_qa_cache(f"{session_id}:{question}", result)
        logger.info("ask() completed successfully for session_id=%s", session_id)
        return result

    @log_call(logger)
    def reset_conversation(self, session_id: str = "default") -> None:
        logger.info("Resetting conversation memory for session_id=%s", session_id)
        self.memory.clear_history(session_id)


# Alias for backward compatibility
RepoBAAgent = RepoBAChatbot


def main() -> None:
    parser = argparse.ArgumentParser(description="Repository BA Chatbot with Neo4j Graph RAG and Redis Memory")
    parser.add_argument("question", nargs="?", default=None, help="Question to ask (interactive if omitted)")
    parser.add_argument("--session-id", default="default", help="Session ID for Redis conversation memory")
    parser.add_argument("--uri", default=None, help="Neo4j URI")
    parser.add_argument("--username", default=None, help="Neo4j username")
    parser.add_argument("--password", default=None, help="Neo4j password")
    parser.add_argument("--database", default=None, help="Neo4j database")
    parser.add_argument("--redis-url", default=None, help="Redis URL")
    args = parser.parse_args()

    bot = RepoBAChatbot(
        uri=args.uri,
        username=args.username,
        password=args.password,
        database=args.database,
        redis_url=args.redis_url,
    )

    if args.question:
        try:
            result = bot.ask(args.question, session_id=args.session_id)
            print(f"\nAssistant: {result['answer']}")
        finally:
            bot.close()
        return

    print("Repository BA Chatbot (Neo4j Graph RAG + Redis Memory)")
    print("Ask repository questions. Commands: /reset, /exit")

    try:
        while True:
            try:
                question = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye.")
                break

            if question.lower() in {"/exit", "exit", "quit"}:
                break

            if question.lower() == "/reset":
                bot.reset_conversation(session_id=args.session_id)
                print("Conversation history cleared.")
                continue

            try:
                result = bot.ask(question, session_id=args.session_id)
                print(f"\nAssistant: {result['answer']}")
            except Exception as exc:
                print(f"Request failed: {exc}")
    finally:
        bot.close()


# if __name__ == "__main__":
#     main()

