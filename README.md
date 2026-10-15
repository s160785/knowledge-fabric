# Code Knowledge Extraction

Extract knowledge from a source-code repository and return it in a Serenity response. The extracted knowledge is intended for downstream consumers; ContextOS Knowledge Fabric (KF) modules own its storage and retrieval.

## Input

- A code repository to analyze.

The extractor reads the repository and derives knowledge from its source, such as symbols, relationships, dependencies, architecture, and behavior. Results should be grounded in repository evidence so consumers can trace conclusions back to the code.

## Response modes

Both modes use Serenity as the response format. They differ in scope and intended consumer:

1. **Full knowledge (batch):** Return the comprehensive knowledge extracted from the repository for batch processing and downstream ingestion.
2. **Coding agent interface:** Return the relevant, focused knowledge for a coding agent working with the repository, shaped for agent consumption rather than as a full-repository dump.

## Responsibility boundary

This component is responsible for analyzing the input repository and producing the selected Serenity response. It does not define or implement how extracted knowledge is stored, indexed, or retrieved; those responsibilities belong to the ContextOS Knowledge Fabric modules.

## Project layout

```text
src/conseek_knowledge_fabric/
    app/
        main.py         FastAPI routes and server entry point
        schemas.py      Request validation
    utils/
        graph_pipeline.py Repository-to-Neo4j workflow
        browser_views.py  Neo4j Browser view links
    static/
        index.html     Local graph-builder interface
    views/
        full.py        Full-knowledge projection
        coding_agent.py Coding-agent projection
    analysis.py       Tree-sitter extraction
    ast_visualizer.py Interactive standalone AST HTML viewer
    ast_cli.py        AST viewer command line
    clone.py          Git access check and shallow clone
    models.py         Canonical graph and source evidence
    neo4j_store.py    Idempotent Neo4j graph persistence
    scip.py           Optional SCIP index enrichment
    security.py       Dependency, source, malware, and build checks
    serenity.py       Serenity response envelope
    cli.py            Command-line interface
main.py              Root launcher (python main.py)
tests/
```

Set up the package and run its tests with:

```sh
python -m pip install -e .
python -m unittest discover -s tests
```

Run the local FastAPI ingestion UI with:

```sh
conseek-web
```

The root launcher works too: `python main.py`.

Application logs use `DATETIME:THREAD_NO:LEVEL:FUNC:MESSAGE` and cover API calls, branch validation, remote access, clone/switch, repository audits, Tree-sitter extraction, Neo4j writes, and Browser URL construction. Set `CONSEEK_LOG_LEVEL` to `DEBUG`, `INFO`, `WARNING`, or `ERROR` (default `INFO`). Call arguments and return values are not logged, so credentials and repository contents are not included.

Open `http://127.0.0.1:8000`. Enter a local path or Git URL, an optional branch, and Neo4j credentials, build the graph, then select a projection to open its repository-scoped query in Neo4j Browser. For remote URLs, the selected branch is verified with `git ls-remote` and shallow-cloned. For local paths, it runs `git switch`; Git will refuse if local changes conflict, and the app never forces or discards them. Leave the branch blank to use the default branch/current checkout. Set the Browser HTTP URL separately from the Bolt URI for custom ports or remote instances. The UI binds to localhost and does not save or return the password; `.env` can also provide Git clone credentials.

Extract a repository and write a full response to stdout:

```sh
conseek-extract ./path/to/repository
```

Generate an interactive AST viewer for a source file:

```sh
conseek-ast ./path/to/repository src/main.py --output ast-view.html
```

To persist extracted files, symbols, and relationships to Neo4j, set the connection values in the environment or `.env` and opt in with `--neo4j`:

```sh
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USERNAME=neo4j
export NEO4J_PASSWORD=your-local-password
conseek-extract ./path/to/repository --neo4j
```

The loader creates `Repository`, `File`, and `Symbol` nodes, plus `Import` targets and `ExternalSymbol` nodes only for unresolved targets. `DEFINES` and typed `CODE_RELATION` edges preserve file and source relationships. Stable repository-scoped IDs and `MERGE` make repeated imports idempotent. Use `--neo4j-database` or `NEO4J_DATABASE` to select a database. The account needs permission to create constraints and write nodes/relationships.

Open `ast-view.html` in a browser. The page is self-contained and provides node filtering, expand/collapse controls, source ranges, parse-error markers, and selected-node source text. AST exports are capped at 12,000 nodes and input files at 5 MB.

Pass a remote Git URL to shallow-clone it first. Clones go under `.conseek/repos/<repository-name>` by default; use `--clone-root` to change that directory:

```sh
conseek-extract https://github.com/owner/repository.git --clone-root ./repos
```

Before cloning, the CLI runs `git ls-remote` to verify access. For private GitHub/GitLab HTTPS repositories, put `GITHUB_TOKEN`/`GH_TOKEN` or `GITLAB_TOKEN` in the process environment or project-root `.env`. Azure DevOps accepts `AZURE_DEVOPS_EXT_PAT` or `AZURE_DEVOPS_TOKEN`; use `--env-file` to select another file. For other HTTPS Git servers, set `GIT_USERNAME` and `GIT_TOKEN`. Relative `.env` paths are resolved from the project root if the process working directory differs. Credentials are passed through host-scoped temporary Git configuration, never embedded in the URL or command arguments. Git askpass prompts are disabled for clone subprocesses. `.env` and `.conseek/` are ignored by Git.

After resolving the repository, the CLI automatically validates common JSON, TOML, XML, YAML, and Docker build manifests without running build scripts. It also runs OSV Scanner for dependency advisories, Semgrep's `p/security-audit` rules plus local constant-loop review rules, and ClamAV when those tools are installed. Missing scanners are reported as `SKIPPED`, not as clean results. OSV detects known vulnerable dependencies; ClamAV performs signature-based malware scanning. Constant-true-loop matches are heuristics and may include intentional loops.

To include semantic references, generate `index.scip` with a language-specific SCIP indexer and install the SCIP CLI, then pass the index:

```sh
conseek-extract ./path/to/repository --scip-index ./path/to/repository/index.scip
```

The Tree-sitter grammars currently included are Python, JavaScript, TypeScript, Java, Go, and Rust. SCIP enrichment consumes an existing index via `scip print --json`; it does not run an indexer or launch an LSP server. The response envelope is intentionally local to this project until a formal Serenity schema is supplied.


## Overall Flow
flowchart TD

    REPO[Repository]

    REPO --> EXTRACT[Knowledge Extraction]

    EXTRACT --> KG[(Canonical Neo4j KG)]

    KG --> FULL[Full Knowledge View]
    KG --> CODE[Coding Agent View]

    FULL --> S1[Serenity]
    CODE --> S2[Serenity]


## End to End Code Flow:

flowchart TD

    A[Developer Request] --> B[Coding Agent]

    B --> C[Understand Intent]
    C --> D{Repository Knowledge Needed?}

    D -->|Yes| E[ContextOS KF]
    E --> F[Hybrid Retrieval]

    F --> G[Neo4j Vector Search]
    F --> H[Neo4j Graph Traversal]

    G --> I[Relevant Symbols]
    H --> J[Dependencies / Callers / Callees / Impact]

    I --> K[Focused Repository Subgraph]
    J --> K

    K --> L[Serenity Coding Response]

    L --> B

    B --> M[LSP Live Code Intelligence]

    M --> N[Current Definitions]
    M --> O[Current References]
    M --> P[Diagnostics]
    M --> Q[Current Source Locations]

    N --> R[Agent Reasoning]
    O --> R
    P --> R
    Q --> R
    K --> R

    R --> S[Create Modification Plan]

    S --> T[Human Approval?]

    T -->|Required| U[Developer Approval]
    U --> V[Apply Code Changes]

    T -->|Not Required| V

    V --> W[LSP Diagnostics]

    W --> X{Errors?}

    X -->|Yes| R
    X -->|No| Y[Run Tests / Build / Static Analysis]

    Y --> Z{Validation Passed?}

    Z -->|No| R
    Z -->|Yes| AA[Generate Final Serenity Response]

    AA --> AB[Developer]

    V --> AC[Changed Files / Symbols]

    AC --> AD[Incremental KG Update]

    AD --> AE[Neo4j Updated Graph]