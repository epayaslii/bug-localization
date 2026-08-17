import json
import os
import subprocess
import xml.etree.ElementTree as ET

import pytest

from dataset.bench4bl import Bench4BL

TAG = "v1.0"


def _init_gitrepo(gitrepo_path, files):
    """A real, tiny git repo -- Bench4BL reads directly via `git -C gitrepo ls-tree`,
    same offline pattern as dataset/repo_cache.py's bare clones, so a fixture needs a
    real repo, not a mock."""
    os.makedirs(gitrepo_path, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=gitrepo_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=gitrepo_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=gitrepo_path, check=True)
    for rel_path, content in files.items():
        full = os.path.join(gitrepo_path, rel_path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w") as f:
            f.write(content)
        subprocess.run(["git", "add", rel_path], cwd=gitrepo_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "initial"], cwd=gitrepo_path, check=True)
    subprocess.run(["git", "tag", TAG], cwd=gitrepo_path, check=True)


def _bug_xml(bugs):
    """bugs: list of dicts with keys id, summary, description, version, fixed_version,
    bug_type, fixed_files (list of dotted names)."""
    root = ET.Element("bugrepository")
    for b in bugs:
        bug_el = ET.SubElement(root, "bug", id=str(b["id"]), opendate="2020-01-01", fixdate="2020-01-02")
        info = ET.SubElement(bug_el, "buginformation")
        ET.SubElement(info, "summary").text = b.get("summary", "")
        ET.SubElement(info, "description").text = b.get("description", "")
        ET.SubElement(info, "version").text = b.get("version", "")
        ET.SubElement(info, "fixedVersion").text = b.get("fixed_version", "")
        ET.SubElement(info, "type").text = b.get("bug_type", "bug")
        fixed_files = ET.SubElement(bug_el, "fixedFiles")
        for dotted in b.get("fixed_files", []):
            ET.SubElement(fixed_files, "file").text = dotted
    return ET.tostring(root, encoding="unicode")


def _make_project(cache_dir, project, bugs, versions, files, malformed_xml=False):
    proot = cache_dir / project
    os.makedirs(proot / "bugrepo", exist_ok=True)
    with open(proot / "versions.txt", "w") as f:
        json.dump({project: versions}, f)
    with open(proot / "bugrepo" / "repository.xml", "w") as f:
        f.write("<not valid xml" if malformed_xml else _bug_xml(bugs))
    _init_gitrepo(proot / "gitrepo", files)
    return proot


def test_loads_bug_instance_from_real_fixture(tmp_path):
    _make_project(
        tmp_path, "SHAPES",
        bugs=[{
            "id": 1, "summary": "area is wrong", "description": "computeArea returns 0",
            "version": "1.0", "fixed_version": "1.1", "bug_type": "bug",
            "fixed_files": ["org.example.ShapeUtil.java"],
        }],
        versions={"1.0": TAG},
        files={"src/main/java/org/example/ShapeUtil.java": "class ShapeUtil {}\n"},
    )

    bench = Bench4BL(cache_dir=str(tmp_path))
    instances = bench.get_bug_instances()

    assert len(instances) == 1
    bug = instances[0]
    assert bug.repo == "SHAPES"
    assert bug.instance_id == "SHAPES-1"
    assert bug.base_commit == TAG
    assert bug.ground_truths == ["src/main/java/org/example/ShapeUtil.java"]
    assert "area is wrong" in bug.bug_report
    assert "computeArea returns 0" in bug.bug_report
    assert bug.code_files == ["src/main/java/org/example/ShapeUtil.java"]


def test_skips_project_missing_expected_files(tmp_path):
    # A project dir that exists but has none of the three required files/dirs.
    os.makedirs(tmp_path / "INCOMPLETE")

    bench = Bench4BL(cache_dir=str(tmp_path))

    assert bench.get_bug_instances() == []


def test_skips_bug_with_unresolvable_version(tmp_path):
    _make_project(
        tmp_path, "SHAPES",
        bugs=[{
            "id": 1, "summary": "x", "description": "y", "version": "9.9-does-not-exist",
            "fixed_files": ["org.example.ShapeUtil.java"],
        }],
        versions={"1.0": TAG},  # "9.9-does-not-exist" is not a key here
        files={"src/main/java/org/example/ShapeUtil.java": "class ShapeUtil {}\n"},
    )

    bench = Bench4BL(cache_dir=str(tmp_path))

    assert bench.get_bug_instances() == []


def test_skips_bug_with_no_resolvable_fixed_files(tmp_path):
    _make_project(
        tmp_path, "SHAPES",
        bugs=[{
            "id": 1, "summary": "x", "description": "y", "version": "1.0",
            "fixed_files": ["org.example.DoesNotExist.java"],
        }],
        versions={"1.0": TAG},
        files={"src/main/java/org/example/ShapeUtil.java": "class ShapeUtil {}\n"},
    )

    bench = Bench4BL(cache_dir=str(tmp_path))

    assert bench.get_bug_instances() == []


def test_malformed_xml_skips_project_without_crashing(tmp_path):
    _make_project(
        tmp_path, "BROKEN",
        bugs=[], versions={}, files={"a.java": "class A {}\n"},
        malformed_xml=True,
    )

    bench = Bench4BL(cache_dir=str(tmp_path))  # must not raise

    assert bench.get_bug_instances() == []


def test_repo_filter_limits_to_one_project(tmp_path):
    for project in ("SHAPES", "OTHER"):
        _make_project(
            tmp_path, project,
            bugs=[{
                "id": 1, "summary": "x", "description": "y", "version": "1.0",
                "fixed_files": ["org.example.Foo.java"],
            }],
            versions={"1.0": TAG},
            files={"src/main/java/org/example/Foo.java": "class Foo {}\n"},
        )

    bench = Bench4BL(cache_dir=str(tmp_path), repo_filter="SHAPES")
    instances = bench.get_bug_instances()

    assert len(instances) == 1
    assert instances[0].repo == "SHAPES"


def test_resolve_dotted_path_matches_nested_suffix():
    code_files = ["src/main/java/org/example/ShapeUtil.java", "src/test/java/org/example/ShapeUtilTest.java"]
    resolved = Bench4BL._resolve_dotted_path("org.example.ShapeUtil.java", code_files)
    assert resolved == "src/main/java/org/example/ShapeUtil.java"


def test_resolve_dotted_path_returns_none_when_no_match():
    code_files = ["src/main/java/org/example/ShapeUtil.java"]
    assert Bench4BL._resolve_dotted_path("org.example.NoSuchClass.java", code_files) is None


def test_resolve_dotted_path_returns_none_for_non_java_suffix():
    code_files = ["src/main/resources/example.xml"]
    assert Bench4BL._resolve_dotted_path("org.example.example.xml", code_files) is None


def test_get_bug_instances_sample_size_with_seed_is_deterministic(tmp_path):
    _make_project(
        tmp_path, "SHAPES",
        bugs=[
            {"id": i, "summary": f"bug {i}", "description": "d", "version": "1.0",
             "fixed_files": ["org.example.Foo.java"]}
            for i in range(1, 6)
        ],
        versions={"1.0": TAG},
        files={"src/main/java/org/example/Foo.java": "class Foo {}\n"},
    )

    bench = Bench4BL(cache_dir=str(tmp_path))
    first = bench.get_bug_instances(sample_size=2, random_sample=True, random_seed=42)
    second = bench.get_bug_instances(sample_size=2, random_sample=True, random_seed=42)

    assert len(first) == 2
    assert [b.instance_id for b in first] == [b.instance_id for b in second]
