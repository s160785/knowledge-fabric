import importlib
import base64
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml
from fastapi.testclient import TestClient

from conseek_knowledge_fabric.graphify_analysis import extract_repository
from conseek_knowledge_fabric.clone import clone_repository, resolve_repository
from conseek_knowledge_fabric.models import KnowledgeGraph, Relationship, SourceLocation, Symbol
from conseek_knowledge_fabric.neo4j_store import _groups_by_file, persist_graph
from conseek_knowledge_fabric.scip import enrich_from_scip
from conseek_knowledge_fabric.security import RULES, audit_repository
from conseek_knowledge_fabric.views.full import build
from conseek_knowledge_fabric.app.main import app
from conseek_knowledge_fabric.utils.browser_views import browser_view_url
from conseek_knowledge_fabric.utils.logging import ThreadIdFilter, get_logger, log_call


class PackageStructureTests(unittest.TestCase):
    def test_logger_format_includes_timestamp_thread_level_function_and_redacts_args(self):
        import logging

        logger = get_logger("test.structured.logging")
        capture = io.StringIO()
        handler = logging.StreamHandler(capture)
        handler.addFilter(ThreadIdFilter())
        handler.setFormatter(logging.Formatter(
            "%(asctime)s:%(thread_id)s:%(levelname)s:%(function_name)s:%(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        ))
        logger.addHandler(handler)

        @log_call(logger)
        def sample_operation(secret: str) -> None:
            return None

        try:
            sample_operation("never-log-this-token")
        finally:
            logger.removeHandler(handler)
            handler.close()

        lines = capture.getvalue().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertRegex(lines[0], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{4}:Main Thread:INFO:sample_operation:START$")
        self.assertRegex(lines[1], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{4}:Main Thread:INFO:sample_operation:END$")
        self.assertNotIn("never-log-this-token", capture.getvalue())

    def test_public_modules_are_importable(self):
        modules = (
            "conseek_knowledge_fabric.graphify_analysis",
            "conseek_knowledge_fabric.ast_cli",
            "conseek_knowledge_fabric.clone",
            "conseek_knowledge_fabric.models",
            "conseek_knowledge_fabric.serenity",
            "conseek_knowledge_fabric.scip",
            "conseek_knowledge_fabric.neo4j_store",
            "conseek_knowledge_fabric.ba_agent",
            "conseek_knowledge_fabric.app.main",
            "conseek_knowledge_fabric.app.schemas",
            "conseek_knowledge_fabric.utils.browser_views",
            "conseek_knowledge_fabric.utils.graph_pipeline",
            "main",
            "conseek_knowledge_fabric.security",
            "conseek_knowledge_fabric.cli",
            "conseek_knowledge_fabric.views.full",
            "conseek_knowledge_fabric.views.coding_agent",
        )

        for module in modules:
            with self.subTest(module=module):
                importlib.import_module(module)

    def test_repo_ba_agent_uses_neo4j_for_repo_questions(self):
        from conseek_knowledge_fabric.ba_agent import RepoBAAgent

        class FakeSession:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def run(self, query, **kwargs):
                class Result:
                    def data(self):
                        return [{"name": "main", "kind": "Function", "file": "src/app.py"}]

                return Result()

        class FakeDriver:
            def __init__(self):
                self.session = lambda database=None: FakeSession()

        with patch("conseek_knowledge_fabric.ba_agent.GraphDatabase.driver", return_value=FakeDriver()):
            agent = RepoBAAgent("bolt://localhost:7687", "neo4j", "secret")
            response = agent.ask("What are the entry points in this repo?")

        self.assertIn("entry", response["answer"].lower())
        self.assertTrue(response["evidence"])

    def test_extracts_symbols_imports_and_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "app.py").write_text(
                "from helpers import greet\n\n"
                "class Greeter:\n"
                "    def run(self):\n"
                "        return greet('world')\n",
                encoding="utf-8",
            )
            (root / "helpers.py").write_text(
                "def greet(name):\n    return name\n",
                encoding="utf-8",
            )

            graph = extract_repository(root)
            names = {symbol.name for symbol in graph.symbols}
            kinds = {relationship.kind for relationship in graph.relationships}
            response = build(graph)

            self.assertTrue({"Greeter", "run", "greet"}.issubset(names))
            self.assertIn("imports", kinds)
            self.assertIn("calls", kinds)
            self.assertEqual(response["mode"], "full")
            self.assertEqual(len(graph.files), 2)

    def test_scip_index_adds_semantic_references(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "app.py"
            source.write_text("def greet():\n    return greet()\n", encoding="utf-8")
            graph = extract_repository(root)
            index = {
                "metadata": {"projectRoot": root.as_uri()},
                "documents": [{
                    "relativePath": "app.py",
                    "symbols": [{"symbol": "scip-python python 1 greet.", "displayName": "greet", "kind": "Function"}],
                    "occurrences": [
                        {"symbol": "scip-python python 1 greet.", "symbolRoles": 1, "singleLineRange": {"line": 0, "startCharacter": 4, "endCharacter": 9}},
                        {"symbol": "scip-python python 1 greet.", "symbolRoles": 0, "singleLineRange": {"line": 1, "startCharacter": 11, "endCharacter": 16}},
                    ],
                }],
            }
            completed = SimpleNamespace(returncode=0, stdout=json.dumps(index), stderr="")

            with patch("conseek_knowledge_fabric.scip.shutil.which", return_value="scip"), patch(
                "conseek_knowledge_fabric.scip.subprocess.run", return_value=completed
            ):
                enrich_from_scip(graph, root / "index.scip")

            semantic_references = [edge for edge in graph.relationships if edge.source_index == "scip"]
            self.assertEqual(len(semantic_references), 1)
            self.assertEqual(semantic_references[0].kind, "references")

    def test_supported_grammars_extract_function_symbols(self):
        samples = {
            "a.py": "def greet():\n    pass\n",
            "b.js": "function greet() {}",
            "c.ts": "function greet() {}",
            "d.java": "class Demo { void greet() {} }",
            "e.go": "package main\nfunc greet() {}",
            "f.rs": "fn greet() {}",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for filename, source in samples.items():
                (root / filename).write_text(source, encoding="utf-8")

            graph = extract_repository(root)

        self.assertEqual(sum(symbol.name == "greet" for symbol in graph.symbols), len(samples))
        self.assertEqual(graph.diagnostics, [])

    def test_clone_repository_uses_git_without_shell(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("conseek_knowledge_fabric.clone.subprocess.run") as run:
                destination = clone_repository(
                    "https://github.com/example/project.git",
                    Path(directory) / "repos",
                )

        self.assertEqual(destination.name, "project")
        self.assertEqual(run.call_args.args[0][:4], ["git", "clone", "--depth", "1"])
        self.assertTrue(run.call_args.kwargs["check"])
        self.assertNotIn("shell", run.call_args.kwargs)

    def test_remote_branch_is_checked_then_shallow_cloned(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            "conseek_knowledge_fabric.clone.subprocess.run",
            side_effect=[
                SimpleNamespace(returncode=0),
                SimpleNamespace(returncode=0),
                SimpleNamespace(returncode=0),
            ],
        ) as run:
            destination = clone_repository(
                "https://github.com/example/project.git",
                Path(directory) / "repos",
                branch="feature/demo",
            )

        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual(commands[0], ["git", "check-ref-format", "--branch", "feature/demo"])
        self.assertEqual(commands[1], [
            "git", "ls-remote", "--exit-code", "--heads",
            "https://github.com/example/project.git", "refs/heads/feature/demo",
        ])
        self.assertEqual(commands[2], [
            "git", "clone", "--depth", "1", "--branch", "feature/demo", "--single-branch",
            "https://github.com/example/project.git", str(destination),
        ])

    def test_local_branch_switch_is_non_forcing(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            "conseek_knowledge_fabric.clone.subprocess.run",
            side_effect=[SimpleNamespace(returncode=0), SimpleNamespace(returncode=0)],
        ) as run:
            resolved = resolve_repository(directory, branch="feature/demo")

        switch_command = run.call_args.args[0]
        self.assertEqual(resolved, Path(directory).resolve())
        self.assertEqual(switch_command, ["git", "-C", str(resolved), "switch", "--", "feature/demo"])
        self.assertTrue(run.call_args.kwargs["check"])

    def test_invalid_branch_name_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Invalid Git branch name"):
            from conseek_knowledge_fabric.clone import _validate_branch

            _validate_branch("bad branch")

    def test_local_repository_path_is_not_cloned(self):
        with tempfile.TemporaryDirectory() as directory:
            resolved = resolve_repository(directory)
        self.assertEqual(resolved, Path(directory).resolve())

    def test_git_access_preflight_uses_dotenv_token_without_leaking_it_to_args(self):
        with patch("conseek_knowledge_fabric.clone.dotenv_values", return_value={"GITHUB_TOKEN": "secret-token"}), patch(
            "conseek_knowledge_fabric.clone.subprocess.run"
        ) as run:
            from conseek_knowledge_fabric.clone import check_repository_access

            check_repository_access("https://github.com/example/private.git", ".env")

        command = run.call_args.args[0]
        environment = run.call_args.kwargs["env"]
        self.assertEqual(command, ["git", "ls-remote", "https://github.com/example/private.git"])
        self.assertNotIn("secret-token", " ".join(command))
        config = {
            environment[f"GIT_CONFIG_KEY_{index}"]: environment[f"GIT_CONFIG_VALUE_{index}"]
            for index in range(int(environment["GIT_CONFIG_COUNT"]))
        }
        self.assertEqual(config["credential.interactive"], "false")
        self.assertEqual(config["credential.helper"], "")
        self.assertEqual(
            config["http.https://github.com/.extraheader"],
            "AUTHORIZATION: basic " + base64.b64encode(b"x-access-token:secret-token").decode("ascii"),
        )

    def test_git_dotenv_lookup_is_anchored_to_project_when_cwd_file_is_missing(self):
        from conseek_knowledge_fabric.clone import _git_environment

        with patch("conseek_knowledge_fabric.clone.dotenv_values", return_value={}) as load_env:
            _git_environment("https://github.com/example/repo.git", ".env")

        env_path = Path(load_env.call_args.args[0])
        self.assertTrue(env_path.is_absolute())
        self.assertEqual(env_path, Path(__file__).resolve().parents[1] / ".env")

    def test_git_child_environment_disables_vscode_credential_popups(self):
        from conseek_knowledge_fabric.clone import _git_environment

        vscode_env = {
            "GIT_ASKPASS": "/vscode/askpass.sh",
            "SSH_ASKPASS": "/vscode/ssh-askpass.sh",
            "VSCODE_GIT_ASKPASS_MAIN": "/vscode/main.js",
            "VSCODE_GIT_ASKPASS_NODE": "/usr/bin/node",
            "GIT_TERMINAL_PROMPT": "1",
        }
        with patch.dict("os.environ", vscode_env), patch(
            "conseek_knowledge_fabric.clone.dotenv_values", return_value={}
        ):
            environment = _git_environment("https://github.com/example/repo.git", ".env")

        self.assertNotIn("GIT_ASKPASS", environment)
        self.assertNotIn("SSH_ASKPASS", environment)
        self.assertFalse(any(key.startswith("VSCODE_GIT_ASKPASS") for key in environment))
        self.assertEqual(environment["GIT_TERMINAL_PROMPT"], "0")
        self.assertEqual(environment["GCM_INTERACTIVE"], "Never")
        config = {
            environment[f"GIT_CONFIG_KEY_{index}"]: environment[f"GIT_CONFIG_VALUE_{index}"]
            for index in range(int(environment["GIT_CONFIG_COUNT"]))
        }
        self.assertEqual(config["credential.interactive"], "false")
        self.assertEqual(config["credential.helper"], "")
        self.assertEqual(config["core.askPass"], "")

    def test_azure_devops_pat_is_encoded_without_environment_debug_output(self):
        from conseek_knowledge_fabric.clone import _git_environment

        output = io.StringIO()
        with patch("conseek_knowledge_fabric.clone.dotenv_values", return_value={
            "AZURE_DEVOPS_EXT_PAT": "private-pat-value",
        }), contextlib.redirect_stdout(output):
            environment = _git_environment(
                "https://dev.azure.com/org/project/_git/repository",
                ".env",
            )

        self.assertEqual(output.getvalue(), "")
        config = {
            environment[f"GIT_CONFIG_KEY_{index}"]: environment[f"GIT_CONFIG_VALUE_{index}"]
            for index in range(int(environment["GIT_CONFIG_COUNT"]))
        }
        self.assertEqual(
            config["http.https://dev.azure.com/.extraheader"],
            "AUTHORIZATION: basic " + base64.b64encode(b":private-pat-value").decode(),
        )
        self.assertNotIn("private-pat-value", config["http.https://dev.azure.com/.extraheader"])

    def test_credentials_in_remote_url_are_rejected_without_echoing_them(self):
        from conseek_knowledge_fabric.clone import check_repository_access

        with self.assertRaises(ValueError) as error:
            check_repository_access("https://user:private-token@example.com/repo.git")

        self.assertNotIn("private-token", str(error.exception))

    def test_audit_reports_invalid_manifests_and_missing_scanners(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "pyproject.toml").write_text("[build-system\n", encoding="utf-8")
            with patch("conseek_knowledge_fabric.security.shutil.which", return_value=None):
                results = audit_repository(root)

        self.assertTrue(any("invalid build manifest pyproject.toml" in item for item in results))
        self.assertTrue(any("osv-scanner not installed" in item for item in results))
        self.assertTrue(any("semgrep not installed" in item for item in results))
        self.assertTrue(any("clamscan not installed" in item for item in results))

    def test_scanner_findings_are_reported_and_loop_rules_parse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            osv = {"results": [{"packages": [{"vulnerabilities": [{"id": "OSV-1"}]}]}]}
            semgrep = {"results": [{
                "check_id": "python-constant-true-loop",
                "path": "app.py",
                "start": {"line": 3},
                "extra": {"message": "review loop"},
            }]}
            outputs = [
                SimpleNamespace(returncode=1, stdout=json.dumps(osv), stderr=""),
                SimpleNamespace(returncode=0, stdout=json.dumps(semgrep), stderr=""),
                SimpleNamespace(returncode=1, stdout="sample.exe FOUND", stderr=""),
            ]
            with patch("conseek_knowledge_fabric.security.shutil.which", side_effect=lambda name: name), patch(
                "conseek_knowledge_fabric.security.subprocess.run", side_effect=outputs
            ):
                results = audit_repository(root)

        self.assertTrue(any("OSV-1" in item for item in results))
        self.assertTrue(any("python-constant-true-loop" in item for item in results))
        self.assertTrue(any("ClamAV detected" in item for item in results))
        rules = yaml.safe_load(RULES.read_text(encoding="utf-8"))
        self.assertTrue(any("loop" in rule["id"] for rule in rules["rules"]))

    def test_neo4j_mapping_scopes_symbols_and_resolves_local_edges(self):
        location = SourceLocation("app.py", 1, 0, 2, 0)
        module = Symbol("app.py::module", "app", "module", "py", location, "app.py")
        function = Symbol("app.py::function:greet@1:0", "greet", "function", "py", location, "def greet")
        relationship = Relationship(module.id, function.id, "calls", location, "greet()")
        graph = KnowledgeGraph("/repo-a", ["app.py"], [module, function], [relationship])

        symbols, relationships = _groups_by_file(graph)

        self.assertEqual(symbols["app.py"][1]["id"], "/repo-a::app.py::function:greet@1:0")
        self.assertEqual(relationships["app.py"][0]["source"], "/repo-a::app.py::module")
        self.assertEqual(relationships["app.py"][0]["target"], "/repo-a::app.py::function:greet@1:0")
        self.assertTrue(relationships["app.py"][0]["local_target"])

    def test_neo4j_persistence_upserts_files_symbols_and_edges(self):
        location = SourceLocation("app.py", 1, 0, 2, 0)
        module = Symbol("app.py::module", "app", "module", "py", location, "app.py")
        function = Symbol("app.py::function:greet@1:0", "greet", "function", "py", location, "def greet")
        relationship = Relationship(module.id, function.id, "calls", location, "greet()")
        graph = KnowledgeGraph("/repo-a", ["app.py"], [module, function], [relationship])

        class FakeTransaction:
            queries = []

            def run(self, query, **parameters):
                self.queries.append((query, parameters))
                return SimpleNamespace(consume=lambda: None)

        transaction = FakeTransaction()

        class FakeSession:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def run(self, query, **parameters):
                transaction.queries.append((query, parameters))
                return SimpleNamespace(consume=lambda: None)

            def execute_write(self, callback, *args):
                return callback(transaction, *args)

        class FakeDriver:
            def session(self, database=None):
                return FakeSession()

        counts = persist_graph(FakeDriver(), graph)
        queries = [query for query, _parameters in transaction.queries]
        parameters = [params for _query, params in transaction.queries]

        self.assertEqual(counts, {"files": 1, "symbols": 2, "relationships": 1})
        self.assertTrue(any("CREATE CONSTRAINT symbol_id" in query for query in queries))
        self.assertTrue(any("MERGE (file)-[:DEFINES]->(symbol)" in query for query in queries))
        self.assertTrue(any("MATCH (target:Symbol {id: item.target})" in query for query in queries))
        self.assertTrue(any("MERGE (source)-[:CODE_RELATION" in query for query in queries))
        self.assertTrue(any(params.get("symbols") for params in parameters))

    def test_browser_view_url_scopes_query_to_repository(self):
        url = browser_view_url(
            "http://localhost:7474/browser/",
            "/repos/a'b",
            "symbols",
            "neo4j",
        )
        from urllib.parse import parse_qs, urlsplit

        query = parse_qs(urlsplit(url).query)
        self.assertEqual(urlsplit(url).netloc, "localhost:7474")
        self.assertEqual(query["cmd"], ["play"])
        self.assertEqual(query["db"], ["neo4j"])
        self.assertIn("/repos/a\\'b", query["arg"][0])

    def test_web_api_build_uses_pipeline_and_never_returns_password(self):
        payload = {
            "repository": "https://github.com/example/demo.git",
            "branch": "feature/demo",
            "neo4j_uri": "bolt://localhost:7687",
            "neo4j_username": "neo4j",
            "neo4j_password": "not-returned",
            "neo4j_database": "neo4j",
            "neo4j_browser_url": "http://localhost:7474/browser/",
        }
        result = {
            "repository": "/repos/demo",
            "counts": {"files": 1, "symbols": 1, "relationships": 0},
            "diagnostics": ["OK: manifest valid"],
        }
        with patch("conseek_knowledge_fabric.app.main.build_repository_graph", return_value=result) as build_graph:
            response = TestClient(app).post("/api/build", json=payload)

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["counts"]["symbols"], 1)
        self.assertEqual(body["diagnostics"], ["OK: manifest valid"])
        self.assertEqual(set(body["browser_urls"]), {"overview", "files", "symbols", "relationships"})
        self.assertNotIn("not-returned", response.text)
        self.assertEqual(build_graph.call_args.args[3], "not-returned")
        self.assertEqual(build_graph.call_args.args[5], "feature/demo")

    def test_web_home_and_invalid_browser_url(self):
        client = TestClient(app)
        home = client.get("/")
        ba_page = client.get("/ba-chat")
        invalid = client.post("/api/build", json={
            "repository": "/repo",
            "neo4j_password": "secret-value",
            "neo4j_browser_url": "javascript:alert(1)",
        })

        self.assertEqual(home.status_code, 200)
        self.assertIn("Build a navigable code graph", home.text)
        self.assertIn('href="/ba-chat"', home.text)
        self.assertEqual(ba_page.status_code, 200)
        self.assertIn("Sessions", ba_page.text)
        self.assertEqual(invalid.status_code, 422)
        self.assertNotIn("secret-value", invalid.text)