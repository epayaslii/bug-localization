# The production chunk-relevance-judgment prompt, extracted as a named template constant
# (rather than left inline as an f-string) so it can serve as generation-0 of the
# prompt-optimization search (scripts/optimize_relevance_prompt.py) and so a mutated
# variant can be substituted in wholesale via generate_chunk_relevance_feedback_prompt's
# template_override param -- the original, confirmed-negative baseline stays here
# unmodified and in git history regardless of what any search finds.
DEFAULT_CHUNK_RELEVANCE_TEMPLATE = """
        You are a bug localization expert judging code-segment relevance.

        Bug Report:
        Repo: {repo}
        Instance ID: {instance_id}
        Hints: {hints_text}
        Bug Report: {bug_report}

        Below are {num_chunks} code segments (chunks of methods/functions, not whole
        files) from candidate files retrieved for this bug report. For EACH segment, judge
        whether it is plausibly relevant to fixing this bug (relevant=true) or not
        (relevant=false). Judge every segment listed -- do not skip any. A file can have
        multiple segments; judge each independently based on its own content, not the
        file's other segments. Return each judgment with the exact file path and chunk
        index shown.

        Code Segments:
        {chunks_text}

        Return a judgment for every one of the {num_chunks} segments listed above, using
        their exact file paths and chunk indices.
        """

# Placeholders DEFAULT_CHUNK_RELEVANCE_TEMPLATE.format(...) is always called with -- a
# mutated variant that drops or renames one of these will raise KeyError at format time
# (scripts/optimize_relevance_prompt.py treats that as a failed/zero-scoring variant, not
# a crash), so the mutation meta-prompt is explicit about keeping them intact.
CHUNK_RELEVANCE_TEMPLATE_PLACEHOLDERS = [
    "repo", "instance_id", "hints_text", "bug_report", "num_chunks", "chunks_text",
]

# Sparse-output alternative to DEFAULT_CHUNK_RELEVANCE_TEMPLATE (2026-08-21): asks the
# model to list only the chunks it judges relevant, instead of a verdict for every one.
# A dense response spends most of its output-token budget restating "not relevant" for
# the large majority of a candidate pool -- confirmed on a real dev-set instance to leave
# 174/211 chunks unjudged (silently truncated by max_tokens) even at a raised context
# window. This format needs far fewer output tokens for the same information, since most
# chunks are never mentioned at all. Pairs with SparseChunkRelevanceResponse in
# method/models.py, not ChunkRelevanceFeedbackResponse.
DEFAULT_SPARSE_CHUNK_RELEVANCE_TEMPLATE = """
        You are a bug localization expert judging code-segment relevance.

        Bug Report:
        Repo: {repo}
        Instance ID: {instance_id}
        Hints: {hints_text}
        Bug Report: {bug_report}

        Below are {num_chunks} code segments (chunks of methods/functions, not whole
        files) from candidate files retrieved for this bug report. Most of these segments
        are NOT relevant to this bug -- your job is to find the few that ARE.

        List ONLY the segments you judge relevant to fixing this bug, using their exact
        file path and chunk index. Do NOT list segments you judge not relevant -- any
        segment you omit is treated as not relevant. If none are relevant, return an
        empty list. A file can have multiple segments; judge each independently based on
        its own content, not the file's other segments.

        Code Segments:
        {chunks_text}
        """

SPARSE_CHUNK_RELEVANCE_TEMPLATE_PLACEHOLDERS = [
    "repo", "instance_id", "hints_text", "bug_report", "num_chunks", "chunks_text",
]


class PromptGenerator:

    def generate_openai_prompt(self, bug, code_files_chunk=None):
        """
        Generate prompt for bug localization with optional code file chunk.
        
        Args:
            bug: BugInstance object
            code_files_chunk: Optional list of code files for this chunk
        """
        code_files_text = ""
        if code_files_chunk:
            code_files_text = f"\n\nCode Files in this chunk:\n" + "\n".join(f"- {file}" for file in code_files_chunk)
        elif hasattr(bug, 'code_files') and bug.code_files:
            code_files_text = f"\n\nCode Files:\n" + "\n".join(f"- {file}" for file in bug.code_files)
        
        prompt = f"""
        You are a bug localization expert. Given a bug report and a list of Python files in a repository, 
        your task is to identify which files are most likely related to the reported bug. You will
        return 10 files that are most likely related to the reported bug.

        Instructions:
        - Analyze the bug description.
        - Consider the file names and their potential relevance.
        - Return a ranked list of 10 Python files based on the probability that they contain the bug.
        - If you're analyzing a chunk of files, focus only on the files provided in this chunk.

        Bug Report:
        Repo: {bug.repo}
        Instance ID: {bug.instance_id}
        Base Commit: {bug.base_commit}
        Hints: {bug.hints_text}
        Bug Report: {bug.bug_report}
        Code Files:
        {code_files_text}

        """

        return prompt

    def generate_chunk_aggregation_prompt(self, bug, chunk_responses):
        """
        Generate prompt to aggregate responses from multiple chunks.
        
        Args:
            bug: BugInstance object
            chunk_responses: List of responses from each chunk
        """
        responses_text = "\n\n".join([
            f"Chunk {i+1} Analysis:\n{response}" 
            for i, response in enumerate(chunk_responses)
        ])
        
        prompt = f"""
        You are a bug localization expert. You have analyzed a large codebase in chunks and received 
        individual analysis results for each chunk. Your task is to aggregate these results into a 
        final ranked list of files most likely to contain the bug.

        Instructions:
        - Review all chunk analyses below
        - Consider the confidence and reasoning from each chunk
        - Merge and rank all suggested files into a single prioritized list
        - Remove duplicates and consolidate similar recommendations
        - Return each entry as the bare file path only (e.g. "src/foo/bar.py") in
          rank order, most likely first. Do NOT include numbering, prefixes,
          confidence scores, or any other text alongside the path.

        Original Bug Report:
        Repo: {bug.repo}
        Instance ID: {bug.instance_id}
        Hints: {bug.hints_text}
        Bug Report: {bug.bug_report}

        Chunk Analysis Results:
        {responses_text}

        Please provide a final aggregated ranking of the most likely files containing the bug:
        """
        
        return prompt
    
    def generate_relevance_feedback_prompt(self, bug, candidate_files, contents, snippet_chars=800):
        """Batched relevance-feedback prompt (BRaIn-style, scoped to file granularity
        rather than per-code-segment -- see docs/relevance_feedback_scoping.md). One call
        judges every candidate at once instead of one call per candidate, keeping this
        close to the project's existing ~1-call-per-bug cost profile.
        """
        files_text = "\n\n".join(
            f"[{i + 1}] {path}\n```\n{(contents.get(path) or '(content unavailable)')[:snippet_chars]}\n```"
            for i, path in enumerate(candidate_files)
        )

        prompt = f"""
        You are a bug localization expert judging code-file relevance.

        Bug Report:
        Repo: {bug.repo}
        Instance ID: {bug.instance_id}
        Hints: {bug.hints_text}
        Bug Report: {bug.bug_report}

        Below are {len(candidate_files)} candidate files retrieved for this bug report, each
        with a short content snippet. For EACH file, judge whether it is plausibly relevant
        to fixing this bug (relevant=true) or not (relevant=false). Judge every file listed
        -- do not skip any. Base your judgment on the actual code shown, not just the file
        name.

        Candidate Files:
        {files_text}

        Return a judgment for every one of the {len(candidate_files)} files listed above,
        using their exact paths.
        """

        return prompt

    def generate_chunk_relevance_feedback_prompt(self, bug, chunks, snippet_chars=800, template_override=None):
        """Chunk-granularity counterpart to generate_relevance_feedback_prompt -- judges
        method/segment-level chunks instead of whole files, matching what BRaIn (JavaParser
        method segmentation) and IQLoc (per-method cross-encoder scoring) actually validated,
        rather than this project's original file-level scoping-down (see
        docs/relevance_feedback_scoping.md's "Open questions" and the file-vs-chunk
        discussion that followed). `chunks` is a list of (file, chunk_index, chunk_text)
        tuples, e.g. from method.embedding_retriever._chunk_file_content per candidate file.
        Still one batched call for every chunk across every candidate, not one call per
        chunk, to keep this at the project's ~1-call-per-bug cost profile.

        `template_override`: a format-string using the same placeholders as
        DEFAULT_CHUNK_RELEVANCE_TEMPLATE (repo, instance_id, hints_text, bug_report,
        num_chunks, chunks_text) -- lets scripts/optimize_relevance_prompt.py swap in a
        mutated instruction template without touching this function's own logic. None
        (the default) reproduces the original, confirmed-negative production prompt exactly.
        """
        chunks_text = "\n\n".join(
            f"[{i + 1}] {path} (chunk {chunk_index})\n```\n{chunk_text[:snippet_chars]}\n```"
            for i, (path, chunk_index, chunk_text) in enumerate(chunks)
        )

        template = template_override or DEFAULT_CHUNK_RELEVANCE_TEMPLATE
        return template.format(
            repo=bug.repo, instance_id=bug.instance_id, hints_text=bug.hints_text,
            bug_report=bug.bug_report, num_chunks=len(chunks), chunks_text=chunks_text,
        )

    def generate_sparse_chunk_relevance_feedback_prompt(self, bug, chunks, snippet_chars=800, template_override=None):
        """Sparse-output alternative to generate_chunk_relevance_feedback_prompt -- asks
        for only the relevant chunks, not a verdict for every one (see
        DEFAULT_SPARSE_CHUNK_RELEVANCE_TEMPLATE's docstring for why). Pair with
        method.models.SparseChunkRelevanceResponse, not ChunkRelevanceFeedbackResponse.
        Same chunk-tuple input shape as generate_chunk_relevance_feedback_prompt.
        """
        chunks_text = "\n\n".join(
            f"[{i + 1}] {path} (chunk {chunk_index})\n```\n{chunk_text[:snippet_chars]}\n```"
            for i, (path, chunk_index, chunk_text) in enumerate(chunks)
        )

        template = template_override or DEFAULT_SPARSE_CHUNK_RELEVANCE_TEMPLATE
        return template.format(
            repo=bug.repo, instance_id=bug.instance_id, hints_text=bug.hints_text,
            bug_report=bug.bug_report, num_chunks=len(chunks), chunks_text=chunks_text,
        )

    def generate_prompt_mutation_meta_prompt(self, current_template, failure_examples):
        """Meta-prompt for method.prompt_optimizer.generate_prompt_variant -- asks an LLM
        to rewrite the chunk-relevance-judgment instruction template given a handful of
        the current template's actual mistakes on the dev set (BRaIn/SAMMO-style mutation,
        hand-rolled rather than the SAMMO library itself since MN5 has no internet to
        install a new package and this loop must run identically local/MN5).

        `failure_examples`: list of dicts with keys `chunk_text` (truncated), `predicted`,
        `actual` -- a mix of false positives and false negatives from scoring
        `current_template` against the dev set, so the mutation is grounded in real errors
        rather than guessing blind.
        """
        examples_text = "\n\n".join(
            f"- Chunk (truncated): {ex['chunk_text'][:300]}\n"
            f"  Model judged: relevant={ex['predicted']}, actually: relevant={ex['actual']}"
            for ex in failure_examples
        )
        placeholders_text = ", ".join(f"{{{p}}}" for p in CHUNK_RELEVANCE_TEMPLATE_PLACEHOLDERS)

        return f"""
        You are optimizing a prompt template used to judge whether a code segment is
        relevant to a bug report (a binary relevant/not-relevant classification task).

        Current template:
        ---
        {current_template}
        ---

        This template made these specific mistakes on real examples:
        {examples_text}

        Rewrite the template's instructional wording to reduce mistakes like these. You
        may change tone, add clarifying criteria, add examples, or restructure the
        instructions -- but the template MUST still be a valid Python str.format() string
        containing every one of these exact placeholders, unchanged: {placeholders_text}.
        Do not remove, rename, or duplicate any of them. Do not add new placeholders.
        Return the complete rewritten template (not just the changed part) plus a short
        rationale for what you changed and why.
        """

    def generate_openai_report_summarizer_prompt(self, bug):
        prompt = f"""
        You are an expert in summarizing bug reports. Given a bug report, your task is to summarize the bug report in a concise manner while preserving all critical information needed for bug localization.

        Instructions:
        - Keep all technical details, error messages, and stack traces
        - Preserve file names, function names, and line numbers mentioned
        - Maintain the core issue description and expected vs actual behavior
        - Remove only redundant information and verbose explanations
        - Ensure the summary is under 3000 tokens while retaining diagnostic value

        Bug Report:
        {bug.bug_report}

        Please provide a concise summary:
        """

        return prompt
    
    