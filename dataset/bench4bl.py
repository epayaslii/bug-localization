"""Bench4BL loader (github.com/exatoa/Bench4BL, ISSTA 2018 reproducibility study).

Each project's downloaded archive already contains a pre-generated bug-repository XML
(`bugrepo/repository.xml`) and a real git repo (`gitrepo/`) with tags matching
`versions.txt` -- the project's own legacy Python 2 pipeline (GitInflator,
BugRepositoryMaker) already did the JIRA-linking/reformatting once when the archive was
built, so this loader reads that output directly rather than depending on the legacy
toolchain at runtime. See scripts/mirror_bench4bl.py for the download+extract step.

Expects a cache dir laid out as <cache_dir>/<PROJECT>/{bugrepo/repository.xml,
versions.txt, gitrepo/} per project, matching each archive's own internal structure.
"""

import json
import logging
import os
import random
import subprocess
import xml.etree.ElementTree as ET

from dataset.base import BugLocalizationDataset
from dataset.models import BugInstance
from dataset.utils import calculate_dataset_token_stats

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench4bl_cache")


class Bench4BL(BugLocalizationDataset):
    def __init__(self, cache_dir=None, repo_filter=None):
        super().__init__()
        self.cache_dir = cache_dir or os.environ.get("BENCH4BL_CACHE_DIR", DEFAULT_CACHE_DIR)
        self.repo_filter = repo_filter
        self._bug_instances = []
        self._file_list_cache = {}  # (gitrepo, commit) -> [paths], avoids re-listing per bug
        self.load_data()

    def _projects(self):
        if not os.path.isdir(self.cache_dir):
            logger.warning(f"Bench4BL cache dir not found: {self.cache_dir}")
            return []
        names = sorted(os.listdir(self.cache_dir))
        if self.repo_filter:
            names = [n for n in names if n == self.repo_filter]
        return [n for n in names if os.path.isdir(os.path.join(self.cache_dir, n))]

    def load_data(self):
        projects = self._projects()
        for project in projects:
            self._load_project(project)
        self.repos = list(set(bug.repo for bug in self._bug_instances))
        logger.info(f"Bench4BL: loaded {len(self._bug_instances)} bug instances from {len(projects)} project(s)")

    def get_token_statistics(self, model="gpt-4o", sample_size=1000):
        """Same shape as SWEBench/BeetleBox's version -- see dataset/swebench.py for the
        canonical implementation this mirrors."""
        if not self._bug_instances:
            self.get_bug_instances()
        return calculate_dataset_token_stats(self._bug_instances, model, sample_size=sample_size)

    def _load_project(self, project):
        proot = os.path.join(self.cache_dir, project)
        repo_xml = os.path.join(proot, "bugrepo", "repository.xml")
        versions_path = os.path.join(proot, "versions.txt")
        gitrepo = os.path.join(proot, "gitrepo")
        if not (os.path.isfile(repo_xml) and os.path.isfile(versions_path) and os.path.isdir(gitrepo)):
            logger.warning(f"Bench4BL: {project} missing expected files, skipping")
            return

        with open(versions_path) as f:
            versions = json.load(f).get(project, {})

        try:
            tree = ET.parse(repo_xml)
        except ET.ParseError as e:
            logger.warning(f"Bench4BL: {project} repository.xml failed to parse: {e}")
            return

        before = len(self._bug_instances)
        for bug_el in tree.getroot().findall("bug"):
            try:
                instance = self._parse_bug(project, gitrepo, versions, bug_el)
                if instance:
                    self._bug_instances.append(instance)
            except Exception as e:
                logger.warning(f"Bench4BL: {project} bug {bug_el.get('id')} failed to process: {e}")
        logger.debug(f"Bench4BL: {project} contributed {len(self._bug_instances) - before} usable instances")

    def _parse_bug(self, project, gitrepo, versions, bug_el):
        info = bug_el.find("buginformation")
        if info is None:
            return None
        summary = (info.findtext("summary") or "").strip()
        description = (info.findtext("description") or "").strip()
        version_label = (info.findtext("version") or "").strip()
        fixed_version_label = (info.findtext("fixedVersion") or "").strip()
        bug_type = (info.findtext("type") or "").strip()

        tag = versions.get(version_label)
        if not tag:
            return None  # version not resolvable to a known git tag -- not usable

        # `gitrepo` is a real git repo transferred with the archive; git accepts tag names
        # anywhere a commit-ish is expected, so no separate SHA lookup is needed. Same
        # "no live network fetch" pattern as dataset/repo_cache.py's bare-clone mirrors.
        base_commit = tag
        code_files = self._list_files_at_commit(gitrepo, base_commit)
        if not code_files:
            return None

        fixed_file_paths = [f.text.strip() for f in bug_el.findall("fixedFiles/file") if f.text and f.text.strip()]
        ground_truths = []
        for dotted in fixed_file_paths:
            resolved = self._resolve_dotted_path(dotted, code_files)
            if resolved:
                ground_truths.append(resolved)
        if not ground_truths:
            return None  # nothing resolvable to a real path -- not usable as a localization target

        bug_report = f"Summary: {summary}\n\nDescription:\n{description}"
        instance_id = f"{project}-{bug_el.get('id')}"

        return BugInstance(
            repo=project,
            instance_id=instance_id,
            base_commit=base_commit,
            patch=f"Fixed in version: {fixed_version_label}",
            hints_text=f"Type: {bug_type}\nOpen date: {bug_el.get('opendate', '')}\nFix date: {bug_el.get('fixdate', '')}",
            ground_truths=ground_truths,
            bug_report=bug_report,
            code_files=code_files,
        )

    def _list_files_at_commit(self, gitrepo, commit):
        key = (gitrepo, commit)
        if key in self._file_list_cache:
            return self._file_list_cache[key]
        try:
            out = subprocess.run(
                ["git", "-C", gitrepo, "ls-tree", "-r", "--name-only", commit],
                capture_output=True, text=True, check=True,
            )
        except subprocess.CalledProcessError as e:
            logger.warning(f"Bench4BL: git ls-tree failed for {gitrepo}@{commit}: {e.stderr}")
            self._file_list_cache[key] = []
            return []
        files = [line for line in out.stdout.splitlines() if line.endswith(".java")]
        self._file_list_cache[key] = files
        return files

    @staticmethod
    def _resolve_dotted_path(dotted, code_files):
        """Bench4BL's fixedFiles entries are dotted Java package+class names
        (org.foo.Bar.java), not real file paths -- Maven multi-module projects nest the
        actual path under a module-specific src root, so the dotted form is only a *suffix*
        of the real path. Same tractable dot-to-slash approximation used elsewhere in this
        project for import-graph resolution: convert to a path suffix, then find any file in
        the tree ending with it. Ambiguous matches (multiple files sharing a suffix, rare)
        take the first match rather than failing the whole bug.
        """
        if not dotted.endswith(".java"):
            return None
        suffix = dotted[: -len(".java")].replace(".", "/") + ".java"
        for f in code_files:
            if f == suffix or f.endswith("/" + suffix):
                return f
        return None

    def get_bug_instances(self, sample_size=None, random_sample=False, random_seed=None, exclude_instance_ids=None):
        instances = self._bug_instances
        if exclude_instance_ids:
            instances = [b for b in instances if b.instance_id not in exclude_instance_ids]
        if sample_size is not None and sample_size < len(instances):
            if random_sample:
                if random_seed is not None:
                    random.seed(random_seed)
                return random.sample(instances, sample_size)
            return instances[:sample_size]
        return instances
