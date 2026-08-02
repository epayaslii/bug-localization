import os
import subprocess
from dataset.utils import get_logger

logger = get_logger(__name__)

DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "repo_cache")


def _cache_dir(cache_dir=None):
    return cache_dir or os.getenv("REPO_CACHE_DIR", DEFAULT_CACHE_DIR)


def _bare_repo_path(repo, cache_dir=None):
    safe_name = repo.replace("/", "__") + ".git"
    return os.path.join(_cache_dir(cache_dir), safe_name)


def is_repo_cached(repo, cache_dir=None):
    return os.path.isdir(_bare_repo_path(repo, cache_dir))


def mirror_repo(repo, cache_dir=None):
    """Bare-clone a repo into the local cache, or fetch updates if already cloned."""
    path = _bare_repo_path(repo, cache_dir)
    os.makedirs(_cache_dir(cache_dir), exist_ok=True)

    if os.path.isdir(path):
        logger.info(f"Repo already cached, fetching updates: {repo}")
        subprocess.run(
            ["git", "--git-dir", path, "fetch", "--all", "--tags"],
            check=True, capture_output=True, text=True
        )
    else:
        logger.info(f"Cloning {repo} (bare)...")
        url = f"https://github.com/{repo}.git"
        subprocess.run(
            ["git", "clone", "--bare", url, path],
            check=True, capture_output=True, text=True
        )
    return path


def _ensure_commit(repo, commit_hash, cache_dir=None):
    """If a commit isn't reachable from any fetched ref, try fetching it directly by SHA."""
    path = _bare_repo_path(repo, cache_dir)
    check = subprocess.run(
        ["git", "--git-dir", path, "cat-file", "-e", commit_hash],
        capture_output=True, text=True
    )
    if check.returncode != 0:
        subprocess.run(
            ["git", "--git-dir", path, "fetch", "origin", commit_hash],
            check=True, capture_output=True, text=True
        )


def get_code_files_local(repo, commit_hash, extensions, cache_dir=None):
    path = _bare_repo_path(repo, cache_dir)
    _ensure_commit(repo, commit_hash, cache_dir)
    result = subprocess.run(
        ["git", "--git-dir", path, "ls-tree", "-r", "--name-only", commit_hash],
        check=True, capture_output=True, text=True
    )
    return [p for p in result.stdout.splitlines() if p.endswith(extensions)]


def get_file_content_local(repo, commit_hash, path_in_repo, cache_dir=None):
    path = _bare_repo_path(repo, cache_dir)
    result = subprocess.run(
        ["git", "--git-dir", path, "show", f"{commit_hash}:{path_in_repo}"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise FileNotFoundError(
            f"{path_in_repo}@{commit_hash} not found in local cache for {repo}: {result.stderr.strip()}"
        )
    return result.stdout
