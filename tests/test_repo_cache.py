import pytest

from dataset.repo_cache import (
    _bare_repo_path,
    is_repo_cached,
    get_code_files_local,
    get_file_content_local,
    get_file_contents_batch,
)

# django/django is already mirrored in this repo's local repo_cache/ from earlier
# manual testing in this session -- these tests skip gracefully if it isn't present
# (e.g. a fresh checkout that hasn't run scripts/mirror_repos.py yet).
DJANGO_REPO = "django/django"
DJANGO_COMMIT = "eb16c7260e573ec513d84cb586d96bdf508f3173"

requires_mirrored_django = pytest.mark.skipif(
    not is_repo_cached(DJANGO_REPO),
    reason="django/django is not mirrored in repo_cache/ -- run scripts/mirror_repos.py first",
)


def test_bare_repo_path_replaces_slash_and_appends_git_suffix():
    path = _bare_repo_path("org/repo-name")
    assert path.endswith("org__repo-name.git")


def test_is_repo_cached_false_for_unmirrored_repo():
    assert is_repo_cached("this-repo-is-definitely-not-mirrored-xyz/nope") is False


@requires_mirrored_django
def test_get_code_files_local_returns_python_files():
    files = get_code_files_local(DJANGO_REPO, DJANGO_COMMIT, (".py",))
    assert len(files) > 0
    assert all(f.endswith(".py") for f in files)


@requires_mirrored_django
def test_get_file_content_local_reads_a_real_file():
    files = get_code_files_local(DJANGO_REPO, DJANGO_COMMIT, (".py",))
    sample_path = files[0]
    content = get_file_content_local(DJANGO_REPO, DJANGO_COMMIT, sample_path)
    assert isinstance(content, str)
    assert len(content) > 0


@requires_mirrored_django
def test_get_file_content_local_raises_for_missing_path():
    with pytest.raises(FileNotFoundError):
        get_file_content_local(DJANGO_REPO, DJANGO_COMMIT, "this/path/does/not/exist.py")


@requires_mirrored_django
def test_get_file_contents_batch_matches_individual_reads():
    files = get_code_files_local(DJANGO_REPO, DJANGO_COMMIT, (".py",))
    sample = files[:5]
    batch_contents = get_file_contents_batch(DJANGO_REPO, DJANGO_COMMIT, sample)

    for path in sample:
        if path not in batch_contents:
            continue  # binary/undecodable files are legitimately omitted by the batch API
        individual = get_file_content_local(DJANGO_REPO, DJANGO_COMMIT, path)
        assert batch_contents[path] == individual


@requires_mirrored_django
def test_get_file_contents_batch_omits_missing_files_without_raising():
    result = get_file_contents_batch(DJANGO_REPO, DJANGO_COMMIT, ["this/does/not/exist.py"])
    assert result == {}


def test_get_file_contents_batch_empty_paths_returns_empty_dict():
    assert get_file_contents_batch(DJANGO_REPO, DJANGO_COMMIT, []) == {}
