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

    def generate_reasoning_prompt(self, bug, code_files_chunk=None):
        """
        Like generate_openai_prompt, but asks the model to reason about each candidate
        file's relevance to the bug BEFORE producing the final ranked list, instead of
        picking files directly. Intended to run on an already-narrowed candidate set
        (e.g. after BM25 top-k), not the full repo -- reasoning about every file in a
        large repo would blow up both cost and prompt size.
        """
        files = code_files_chunk if code_files_chunk else getattr(bug, 'code_files', [])
        code_files_text = "\n".join(f"- {file}" for file in files)

        prompt = f"""
        You are a bug localization expert. Given a bug report and a shortlist of candidate
        Python files from a repository, your task is to identify which files are most likely
        related to the reported bug.

        Instructions:
        - For EACH file in the candidate list below, write one short, bug-specific sentence
          of reasoning: does this file's name/path/role plausibly relate to the bug's
          symptoms, error messages, or affected functionality, and why (or why not)?
          Reason about what the file likely does and how that connects to the bug -- do not
          just restate the bug report or the file name.
        - Do not skip any file -- every file in the candidate list must get exactly one
          reasoning entry.
        - A test file (e.g. test_foo.py) that exercises the buggy behavior is topically
          related, but the bug itself almost always lives in the implementation file it
          tests, not the test file. When both an implementation file and its own test file
          are plausible candidates, rank the implementation file above the test file unless
          your reasoning gives a specific reason to believe the defect is in the test code
          itself (e.g. the bug report is about a test failing incorrectly, a fixture, or
          test infrastructure).
        - AFTER reasoning about every file, use that reasoning (not just file names) to
          produce a final ranked list of up to 10 files, most likely to contain the bug first.

        Bug Report:
        Repo: {bug.repo}
        Instance ID: {bug.instance_id}
        Base Commit: {bug.base_commit}
        Hints: {bug.hints_text}
        Bug Report: {bug.bug_report}

        Candidate Files:
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
    
    