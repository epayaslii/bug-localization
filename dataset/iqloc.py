"""IQLoc-extended Bench4BL loader (github.com/asifsamir/IQLoc, JSS 2026).

Reads Bench4BLExtended.json directly instead of Bench4BL's repository.xml -- by default
from bench4bl_cache/Bench4BLExtended.json (gitignored, same as the rest of bench4bl_cache/;
copy it there manually or via scripts/mirror_bench4bl.py --iqloc). Each
record's `label` field distinguishes original Bench4BL bugs (label == 1) from the
~1,740 newer bug reports IQLoc's authors added to extend the benchmark (label
absent). Reuses Bench4BL's git tag/file-listing/dotted-path resolution since
sub_project codes (CAMEL, ELY, AMQP, ...) match this repo's existing
bench4bl_cache/ layout.
"""

import json
import logging
import os

from dataset.bench4bl import DEFAULT_CACHE_DIR, Bench4BL
from dataset.models import BugInstance

logger = logging.getLogger(__name__)


class IQLocExtended(Bench4BL):
    def __init__(self, dataset_json_path=None, cache_dir=None, include_extension=True):
        # Resolved independently of super().__init__() (which sets self.cache_dir) because
        # Bench4BL.__init__ calls self.load_data() itself -- our override needs
        # dataset_json_path set before that call happens, so the default can't wait for it.
        resolved_cache_dir = cache_dir or os.environ.get("BENCH4BL_CACHE_DIR", DEFAULT_CACHE_DIR)
        self.dataset_json_path = dataset_json_path or os.path.join(resolved_cache_dir, "Bench4BLExtended.json")
        self.include_extension = include_extension  # False = original-Bench4BL subset only
        super().__init__(cache_dir=cache_dir)

    def load_data(self):
        with open(self.dataset_json_path) as f:
            records = json.load(f)

        for rec in records:
            is_original = rec.get("label") == 1
            if not is_original and not self.include_extension:
                continue
            try:
                instance = self._parse_record(rec)
                if instance:
                    self._bug_instances.append(instance)
            except Exception as e:
                logger.warning(f"IQLocExtended: bug {rec.get('bug_id')} failed to process: {e}")

        self.repos = list(set(bug.repo for bug in self._bug_instances))
        logger.info(f"IQLocExtended: loaded {len(self._bug_instances)} bug instances")

    def _parse_record(self, rec):
        sub_project = rec["sub_project"]
        proot = os.path.join(self.cache_dir, sub_project)
        gitrepo = os.path.join(proot, "gitrepo")
        if not os.path.isdir(gitrepo):
            return None  # no local mirror for this sub_project -- not usable

        # Unlike Bench4BL's repository.xml (whose <version> is a bare label needing
        # versions.txt to resolve to a git tag), IQLoc's `version` field is already the
        # literal git tag -- confirmed across CAMEL/CODEC/ANDROID/HIVE/HBASE's differing
        # tag-naming conventions. git accepts a tag name anywhere a commit-ish is expected,
        # so no lookup table is needed here.
        tag = rec["version"]

        code_files = self._list_files_at_commit(gitrepo, tag)
        if not code_files:
            return None

        ground_truths = []
        for dotted in rec.get("fixed_files", []):
            resolved = self._resolve_dotted_path(dotted, code_files)
            if resolved:
                ground_truths.append(resolved)
        if not ground_truths:
            return None

        bug_report = f"Summary: {rec['bug_title']}\n\nDescription:\n{rec['bug_description']}"
        instance_id = f"{sub_project}-{rec['bug_id']}"

        return BugInstance(
            repo=sub_project,
            instance_id=instance_id,
            base_commit=tag,
            patch=f"Fixed in version: {rec.get('fixed_version', '')}",
            hints_text=f"Source: IQLocExtended (label={rec.get('label')})",
            ground_truths=ground_truths,
            bug_report=bug_report,
            code_files=code_files,
        )