import os
import subprocess
from dataset.utils import get_logger

logger = get_logger(__name__)

DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "repo_cache")
BENCH4BL_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench4bl_cache")


def _cache_dir(cache_dir=None):
    return cache_dir or os.getenv("REPO_CACHE_DIR", DEFAULT_CACHE_DIR)


def _bare_repo_path(repo, cache_dir=None):
    safe_name = repo.replace("/", "__") + ".git"
    return os.path.join(_cache_dir(cache_dir), safe_name)


def _bench4bl_gitrepo_path(repo):
    """Bench4BL's per-project git repos live in bench4bl_cache/<PROJECT>/gitrepo -- a real,
    non-bare working-tree clone transferred with each project's SourceForge archive (see
    dataset/bench4bl.py), keyed by plain project name (e.g. "WEAVER") rather than this
    module's own bare-clone cache above (keyed by a GitHub "owner/repo" string). Checked
    as a fallback below so callers (bm25_retriever.py, embedding_retriever.py) don't need
    to special-case Bench4BL bugs -- without this, is_repo_cached() was always False for
    every Bench4BL repo and every BM25/embedding representation silently degraded to
    path-only tokens regardless of content-parsing logic (see docs/bench4bl_result.md)."""
    base = os.environ.get("BENCH4BL_CACHE_DIR", BENCH4BL_CACHE_DIR)
    return os.path.join(base, repo, "gitrepo")


def is_repo_cached(repo, cache_dir=None):
    if os.path.isdir(_bare_repo_path(repo, cache_dir)):
        return True
    return os.path.isdir(_bench4bl_gitrepo_path(repo))


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


def get_recent_commit_timestamps(repo, commit_hash, max_commits=2000, cache_dir=None):
    """{path: unix_timestamp of its most recent modification reachable from commit_hash}.
    Walks history backward from commit_hash and records each path's FIRST appearance
    (i.e. its most recent touch) -- doesn't need the full history, just the newest commit
    that touched each path, so this stops recording a path once seen once. max_commits is
    a safety cap for repos with very long histories, not a correctness requirement.
    """
    path = _bare_repo_path(repo, cache_dir)
    _ensure_commit(repo, commit_hash, cache_dir)
    result = subprocess.run(
        ["git", "--git-dir", path, "log", f"--max-count={max_commits}",
         "--name-only", "--pretty=format:__COMMIT__%ct", commit_hash],
        capture_output=True, text=True
    )
    timestamps = {}
    current_ts = None
    for line in result.stdout.splitlines():
        if line.startswith("__COMMIT__"):
            current_ts = int(line[len("__COMMIT__"):])
        elif line.strip() and current_ts is not None and line not in timestamps:
            timestamps[line] = current_ts
    return timestamps


def _cat_file_batch(git_dir, commit_hash, paths):
    if not paths:
        return {}

    input_data = "".join(f"{commit_hash}:{p}\n" for p in paths).encode()
    result = subprocess.run(
        ["git", "--git-dir", git_dir, "cat-file", "--batch"],
        input=input_data, capture_output=True
    )

    data = result.stdout
    contents = {}
    pos = 0
    for p in paths:
        newline_idx = data.index(b"\n", pos)
        header = data[pos:newline_idx].decode(errors="replace")
        pos = newline_idx + 1

        parts = header.split()
        # "missing" responses are "<input> missing" -- checking the LAST token (not
        # requiring exactly 2 parts) matters because a path containing a space splits into
        # more tokens, which silently defeated the old `len(parts) == 2` check and fell
        # through to int(parts[2]) crashing on the literal string "missing" (hit for real on
        # a commit-history-surfaced candidate path missing at the bug's specific commit --
        # 2026-08-25).
        if parts and parts[-1] == "missing":
            continue

        size = int(parts[-1])
        content_bytes = data[pos:pos + size]
        pos += size + 1  # skip content plus the trailing newline git cat-file adds

        try:
            contents[p] = content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            continue

    return contents


def get_file_contents_batch(repo, commit_hash, paths, cache_dir=None):
    """Fetch content for many files at one commit using a single `git cat-file --batch`
    subprocess, instead of one `git show` subprocess per file (much faster for large lists).

    Returns {path: content} for files that were readable as UTF-8 text; missing, binary,
    or undecodable files are simply omitted rather than raising. Falls back to Bench4BL's
    own working-tree gitrepo (see _bench4bl_gitrepo_path) when this module's own bare-clone
    cache doesn't have the repo -- same cat-file mechanism, just a different .git dir, and
    no _ensure_commit fetch attempt since Bench4BL's local clone is already fully
    self-contained (never a live network call either way).
    """
    bare_path = _bare_repo_path(repo, cache_dir)
    if os.path.isdir(bare_path):
        _ensure_commit(repo, commit_hash, cache_dir)
        return _cat_file_batch(bare_path, commit_hash, paths)

    gitrepo = _bench4bl_gitrepo_path(repo)
    if os.path.isdir(gitrepo):
        return _cat_file_batch(os.path.join(gitrepo, ".git"), commit_hash, paths)

    return {}
