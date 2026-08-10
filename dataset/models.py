from pydantic import BaseModel
import logging
import sys
from datetime import datetime
from dataset.utils import get_token_count


class LineRange(BaseModel):
    """One unified-diff hunk's line ranges: old_start/old_lines describe the range in the
    before-fix file (the searchable corpus); new_start/new_lines describe the same range in
    the after-fix file. old_lines/new_lines is 0 for pure insertions/deletions respectively,
    matching unified diff's own `@@ -a,b +c,d @@` convention when a count is omitted (implied 1)."""
    old_start: int
    old_lines: int
    new_start: int
    new_lines: int


class BugInstance(BaseModel):
    repo: str
    instance_id: str
    base_commit: str
    patch: str
    hints_text : str
    ground_truths: list[str]
    bug_report: str
    code_files: list[str]
    after_commit: str | None = None
    line_mappings: dict[str, list[LineRange]] = {}
    """Changed line ranges per ground-truth file path, parsed from `patch`'s diff hunks.
    File-level only when empty (e.g. BeetleBox's patch field is a synthetic placeholder,
    not a real diff -- see dataset/beetlebox.py). Currently populated for SWE-Bench only."""

    def to_string(self) -> str:
        return (
            f"Repo: {self.repo}\n"
            f"Base Commit: {self.base_commit}\n"
            f"Hints: {self.hints_text}\n"
            f"Bug Report: {self.bug_report}\n"
            f"Code Files: {', '.join(self.code_files)}"
        )
    
    def get_token_count(self, model: str = "gpt-4o") -> dict:
        """Get token count breakdown for this bug instance."""
        return {
            'bug_report_tokens': get_token_count(self.bug_report, model),
            'hints_tokens': get_token_count(self.hints_text, model),
            'code_files_tokens': get_token_count(', '.join(self.code_files), model),
            'total_prompt_tokens': get_token_count(self.to_string(), model)
        }