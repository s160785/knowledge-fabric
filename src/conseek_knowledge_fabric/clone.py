"""Clone remote repositories before extraction."""

from __future__ import annotations

import base64
import os
import subprocess
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import dotenv_values
from conseek_knowledge_fabric.utils.logging import get_logger, log_call


logger = get_logger(__name__)


@log_call(logger)
def _resolve_env_file(env_file: str | Path | None) -> Path | None:
    if env_file is None:
        return None
    path = Path(env_file).expanduser()
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    return Path(__file__).resolve().parents[2] / path


@log_call(logger)
def _repository_name(source: str) -> str:
    path = urlsplit(source).path if "://" in source else source.rsplit(":", 1)[-1]
    name = path.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git")
    if not name or name in {".", ".."}:
        raise ValueError(f"Cannot determine repository name from {source!r}")
    return name


@log_call(logger)
def is_git_url(source: str) -> bool:
    return source.startswith(("https://", "http://", "ssh://", "git@"))


@log_call(logger)
def _git_environment(source: str, env_file: str | Path | None) -> dict[str, str]:
    environment = dict(os.environ)
    for key in tuple(environment):
        if key in {"GIT_ASKPASS", "SSH_ASKPASS"} or key.startswith("VSCODE_GIT_ASKPASS"):
            environment.pop(key)
    environment["GIT_TERMINAL_PROMPT"] = "0"
    environment["GCM_INTERACTIVE"] = "Never"
    credentials = {**dotenv_values(_resolve_env_file(env_file or ".env")), **environment}
    parsed = urlsplit(source)
    host = (parsed.hostname or source.split("@", 1)[-1].split(":", 1)[0]).lower()
    token = None
    username = None
    if host == "github.com":
        token = credentials.get("GITHUB_TOKEN") or credentials.get("GIT_TOKEN")
        username = credentials.get("GIT_USERNAME") or "x-access-token"
    elif host == "gitlab.com" or host.endswith(".gitlab.com"):
        token = credentials.get("GITLAB_TOKEN")
        username = "oauth2"
    elif host == "dev.azure.com" or host.endswith(".visualstudio.com"):
        token = (
            credentials.get("AZURE_DEVOPS_EXT_PAT")
            or credentials.get("AZURE_DEVOPS_TOKEN")
            or credentials.get("AZURE_DEVOPS_PAT")
            or credentials.get("GIT_TOKEN")
        )
        username = credentials.get("GIT_USERNAME", "")
    else:
        token = credentials.get("GIT_TOKEN")
        username = credentials.get("GIT_USERNAME")
    git_config = [
        ("credential.interactive", "false"),
        ("credential.helper", ""),
        ("core.askPass", ""),
    ]
    if token:
        authorization = base64.b64encode(f"{username}:{token}".encode()).decode("ascii")
        git_config.append((f"http.https://{host}/.extraheader", f"AUTHORIZATION: basic {authorization}"))
    count = int(environment.get("GIT_CONFIG_COUNT", "0"))
    for offset, (key, value) in enumerate(git_config):
        environment[f"GIT_CONFIG_KEY_{count + offset}"] = key
        environment[f"GIT_CONFIG_VALUE_{count + offset}"] = value
    environment["GIT_CONFIG_COUNT"] = str(count + len(git_config))
    return environment


@log_call(logger)
def _validate_branch(branch: str | None) -> str | None:
    if branch is None:
        return None
    branch = branch.strip()
    if not branch:
        return None
    result = subprocess.run(
        ["git", "check-ref-format", "--branch", branch],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise ValueError(f"Invalid Git branch name: {branch}")
    return branch


@log_call(logger)
def check_repository_access(
    source: str,
    env_file: str | Path | None = None,
    branch: str | None = None,
) -> str | None:
    """Verify that Git can read the remote before attempting a clone."""
    branch = _validate_branch(branch)
    parsed = urlsplit(source)
    if parsed.scheme == "http":
        logger.warning("Rejected insecure HTTP repository URL")
        raise ValueError("Use HTTPS or SSH for remote repository URLs")
    if parsed.username or parsed.password:
        logger.warning("Rejected repository URL with embedded credentials")
        raise ValueError("Do not put credentials in the Git URL; use .env instead")
    try:
        if branch:
            logger.info("Checking requested remote branch")
            command = ["git", "ls-remote", "--exit-code", "--heads", source, f"refs/heads/{branch}"]
        else:
            logger.info("Checking remote repository access")
            command = ["git", "ls-remote", source]
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
            env=_git_environment(source, env_file),
        )
    except subprocess.CalledProcessError as error:
        logger.warning("Remote access check failed exit_code=%s", error.returncode)
        raise PermissionError("Repository is inaccessible; check the URL and Git credentials") from error
    except subprocess.TimeoutExpired as error:
        logger.error("Remote access check timed out")
        raise TimeoutError("Timed out checking repository access") from error
        logger.info("Remote access check completed")
    return branch


@log_call(logger)
def clone_repository(
    source: str,
    clone_root: str | Path = ".conseek/repos",
    env_file: str | Path | None = None,
    branch: str | None = None,
) -> Path:
    """Shallow-clone a repository into clone_root and return its local path."""
    destination = Path(clone_root).expanduser().resolve() / _repository_name(source)
    if destination.exists():
        logger.info("Using existing repository clone")
        return destination
    branch = check_repository_access(source, env_file, branch)
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = ["git", "clone", "--depth", "1"]
    if branch:
        logger.info("Preparing shallow clone for selected branch")
        command.extend(["--branch", branch, "--single-branch"])
    command.extend([source, str(destination)])
    try:
        logger.info("Starting shallow Git clone")
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
            env=_git_environment(source, env_file),
        )
    except subprocess.CalledProcessError as error:
        logger.error("Git clone failed exit_code=%s", error.returncode)
        raise RuntimeError("git clone failed after access check") from error
    except subprocess.TimeoutExpired as error:
        logger.error("Git clone timed out")
        raise TimeoutError("Timed out cloning repository") from error
        logger.info("Git clone completed")
    return destination


@log_call(logger)
def resolve_repository(
    source: str,
    clone_root: str | Path = ".conseek/repos",
    env_file: str | Path | None = None,
    branch: str | None = None,
) -> Path:
    """Return a local repository path, cloning source first when it is a URL."""
    if is_git_url(source):
        return clone_repository(source, clone_root, env_file, branch)
    repository = Path(source).expanduser().resolve(strict=True)
    branch = _validate_branch(branch)
    if branch:
        try:
            logger.info("Switching local checkout to requested branch")
            subprocess.run(
                ["git", "-C", str(repository), "switch", "--", branch],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
                env=_git_environment(str(repository), env_file),
            )
        except subprocess.CalledProcessError as error:
            logger.error("Local branch switch failed exit_code=%s", error.returncode)
            raise RuntimeError(f"Could not switch local repository to branch {branch}; check that it exists and the worktree is clean") from error
        except subprocess.TimeoutExpired as error:
            logger.error("Local branch switch timed out")
            raise TimeoutError("Timed out switching local repository branch") from error
    return repository