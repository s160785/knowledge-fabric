"""Run non-executing repository security and build-manifest checks."""

from __future__ import annotations

import json
import shutil
import subprocess
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml
from conseek_knowledge_fabric.utils.logging import get_logger, log_call


SKIP_DIRS = {".git", ".hg", ".svn", ".venv", "venv", "node_modules", "vendor", "dist", "build"}
RULES = Path(__file__).with_name("security_rules.yml")
logger = get_logger(__name__)


@log_call(logger)
def _run(label: str, command: list[str]) -> tuple[int, str, str]:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 124, "", f"{label} scan timed out"
    except OSError as error:
        return 127, "", f"{label} scan could not start: {error}"


@log_call(logger)
def _osv_scan(root: Path) -> list[str]:
    executable = shutil.which("osv-scanner")
    if not executable:
        return ["SKIPPED: osv-scanner not installed; dependency vulnerabilities not checked"]
    code, stdout, stderr = _run(
        "OSV", [executable, "scan", "source", "-r", str(root), "--format", "json"]
    )
    if code == 127 or code == 124:
        return [f"ERROR: {stderr.strip()}"]
    try:
        report = json.loads(stdout or "{}")
    except json.JSONDecodeError:
        return [f"ERROR: OSV returned invalid JSON: {stderr.strip()[:300]}"]
    if not isinstance(report, dict):
        return ["ERROR: OSV returned an unexpected JSON structure"]
    vulnerabilities = [
        vulnerability
        for result in report.get("results", [])
        for package in result.get("packages", [])
        for vulnerability in (package.get("vulnerabilities") or [])
    ]
    if vulnerabilities:
        ids = sorted({item.get("id", "unknown") for item in vulnerabilities})
        return [f"FINDING: OSV reports {len(vulnerabilities)} dependency vulnerability(s): {', '.join(ids[:20])}"]
    if code:
        return [f"ERROR: OSV scan exited with status {code}: {stderr.strip()[:300]}"]
    return ["OK: OSV found no dependency vulnerabilities"]


@log_call(logger)
def _semgrep_scan(root: Path) -> list[str]:
    executable = shutil.which("semgrep")
    if not executable:
        return ["SKIPPED: semgrep not installed; source security rules not checked"]
    code, stdout, stderr = _run(
        "Semgrep", [executable, "scan", "--config", "p/security-audit", "--config", str(RULES), "--json", str(root)]
    )
    if code == 127 or code == 124:
        return [f"ERROR: {stderr.strip()}"]
    try:
        report = json.loads(stdout or "{}")
    except json.JSONDecodeError:
        return [f"ERROR: Semgrep returned invalid JSON: {stderr.strip()[:300]}"]
    if not isinstance(report, dict):
        return ["ERROR: Semgrep returned an unexpected JSON structure"]
    findings = report.get("results", [])
    summary = report.get("errors", [])
    output = [
        f"FINDING: Semgrep {item.get('check_id', 'rule')} at "
        f"{item.get('path', '?')}:{item.get('start', {}).get('line', '?')}: "
        f"{item.get('extra', {}).get('message', 'review required')}"
        for item in findings
    ]
    if summary:
        output.append(f"ERROR: Semgrep reported {len(summary)} scan error(s)")
    if code not in (0, 1) and not summary:
        output.append(f"ERROR: Semgrep exited with status {code}: {stderr.strip()[:300]}")
    return output or ["OK: Semgrep found no configured source findings"]


@log_call(logger)
def _malware_scan(root: Path) -> list[str]:
    executable = shutil.which("clamscan")
    if not executable:
        return ["SKIPPED: clamscan not installed; malware signatures not checked"]
    code, stdout, stderr = _run(
        "ClamAV", [executable, "--recursive", "--infected", "--no-summary", str(root)]
    )
    if code == 0:
        return ["OK: ClamAV found no infected files"]
    if code == 1:
        detections = [line.strip() for line in stdout.splitlines() if " FOUND" in line]
        return [f"FINDING: ClamAV detected {len(detections)} infected file(s): " + "; ".join(detections[:10])]
    return [f"ERROR: ClamAV exited with status {code}: {(stderr or stdout).strip()[:300]}"]


@log_call(logger)
def _validate_manifests(root: Path) -> list[str]:
    results: list[str] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.is_file() or any(part in SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        relative = path.relative_to(root).as_posix()
        try:
            if path.name in {"pyproject.toml", "Cargo.toml", "poetry.lock", "Cargo.lock"}:
                data = tomllib.loads(path.read_text(encoding="utf-8"))
                if path.name == "pyproject.toml" and "build-system" in data:
                    build = data["build-system"]
                    if not isinstance(build.get("requires"), list) or not build.get("build-backend"):
                        raise ValueError("build-system requires a requires list and build-backend")
                if path.name == "Cargo.toml" and not any(key in data for key in ("package", "workspace")):
                    raise ValueError("expected [package] or [workspace]")
            elif path.name in {"package.json", "package-lock.json", "composer.json", "composer.lock"}:
                data = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    raise ValueError("expected a JSON object")
            elif path.name == "pom.xml":
                ET.parse(path)
            elif path.name == "go.mod":
                content = path.read_text(encoding="utf-8")
                if not any(line.strip().startswith("module ") for line in content.splitlines()):
                    raise ValueError("go.mod is missing module directive")
            elif path.suffix.lower() in {".yaml", ".yml"} and (
                ".github/workflows" in relative
                or path.name in {".gitlab-ci.yml", "docker-compose.yml", "docker-compose.yaml", "compose.yml", "compose.yaml", "Taskfile.yml", "Taskfile.yaml", "azure-pipelines.yml", "action.yml"}
            ):
                if not isinstance(yaml.safe_load(path.read_text(encoding="utf-8")), dict):
                    raise ValueError("expected a YAML mapping")
            elif path.name == "Dockerfile" or path.name.startswith("Dockerfile."):
                content = path.read_text(encoding="utf-8")
                if not any(line.lstrip().upper().startswith("FROM ") for line in content.splitlines()):
                    raise ValueError("Dockerfile is missing FROM instruction")
            else:
                continue
            results.append(f"OK: build manifest {relative} is valid")
        except (OSError, UnicodeError, json.JSONDecodeError, tomllib.TOMLDecodeError, ET.ParseError, yaml.YAMLError, ValueError) as error:
            results.append(f"FINDING: invalid build manifest {relative}: {error}")
    return results or ["INFO: no supported build manifests found for syntax validation"]


@log_call(logger)
def audit_repository(repository: str | Path) -> list[str]:
    """Run available scanners and static build-manifest validation."""
    root = Path(repository).resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    return _validate_manifests(root) + _osv_scan(root) + _semgrep_scan(root) + _malware_scan(root)