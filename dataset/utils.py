import logging
import re
import sys
from datetime import datetime
import tiktoken


def get_diff_hunk(patch: str, path: str) -> str | None:
    """Return the raw diff hunk for `path` from a unified diff, or None if `patch`
    has no real `diff --git` header for it (e.g. BeetleBox's synthetic patch text,
    which is just "Before: <sha>\\nAfter: <sha>").
    """
    escaped = re.escape(path)
    match = re.search(
        rf"diff --git a/{escaped} b/{escaped}\n(.*?)(?=\ndiff --git |\Z)",
        patch, re.DOTALL,
    )
    return match.group(1) if match else None


_HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", re.MULTILINE)


def parse_line_ranges(hunk: str) -> list:
    """Parse a unified-diff hunk's `@@ -a,b +c,d @@` headers into LineRange objects.
    A missing count (bare `-a` / `+c`) means a 1-line range, per unified diff convention."""
    from dataset.models import LineRange

    ranges = []
    for old_start, old_lines, new_start, new_lines in _HUNK_HEADER_RE.findall(hunk):
        ranges.append(LineRange(
            old_start=int(old_start), old_lines=int(old_lines) if old_lines else 1,
            new_start=int(new_start), new_lines=int(new_lines) if new_lines else 1,
        ))
    return ranges


def setup_logging(level=logging.INFO, log_file=None):
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    logging.info("Logging initialized successfully")
    return root_logger

def get_logger(name):
    return logging.getLogger(name)


import requests

def get_code_files(repo, commit_hash, extensions, token=None):
    from dataset.repo_cache import is_repo_cached, get_code_files_local

    if is_repo_cached(repo):
        try:
            return get_code_files_local(repo, commit_hash, extensions)
        except Exception as e:
            logging.warning(f"Local cache read failed for {repo}@{commit_hash}, falling back to GitHub API: {e}")

    url = f"https://api.github.com/repos/{repo}/git/trees/{commit_hash}?recursive=1"
    headers = {}
    if token:
        headers["Authorization"] = f"token {token}"
    r = requests.get(url, headers=headers)
    r.raise_for_status()

    tree = r.json()["tree"]
    selected_files = [item["path"] for item in tree if item["type"] == "blob" and item["path"].endswith(extensions)]
    return selected_files

def is_code_file(path: str, extensions) -> bool:
    """
    Return True if the given path ends with one of the allowed code file extensions.
    Accepts either a tuple of extensions or a single string extension.
    """
    return isinstance(path, str) and path.endswith(extensions)

def filter_code_paths(paths: list[str], extensions) -> list[str]:
    """
    Filter a list of file paths to only those that are code files per the given extensions.
    """
    if not paths:
        return []
    return [p for p in paths if is_code_file(p, extensions)]

def get_token_count(text: str, model: str = "gpt-4") -> int:
    """
    Count tokens in text using tiktoken for the specified model.
    
    Args:
        text: The text to count tokens for
        model: The model name to get the appropriate encoder for
    
    Returns:
        Number of tokens
    """
    try:
        # Map model names to encoders
        encoder_map = {
            "gpt-4": "cl100k_base",
            "gpt-4o": "o200k_base", 
            "gpt-3.5-turbo": "cl100k_base",
        }
        
        encoder_name = encoder_map.get(model, "cl100k_base")
        encoder = tiktoken.get_encoding(encoder_name)
        return len(encoder.encode(text))
    except Exception as e:
        logging.warning(f"Failed to count tokens: {e}")
        # Fallback: rough estimate (1 token ≈ 4 characters)
        return len(text) // 4

def calculate_dataset_token_stats(bug_instances, model="gpt-4o", sample_size=1000):
    """
    Calculate token statistics for a dataset of bug instances.
    
    Args:
        bug_instances: List of BugInstance objects
        model: Model name for token counting
        sample_size: Number of instances to sample for efficient calculation (default: 1000)
    
    Returns:
        Dictionary with token statistics
    """
    import random
    
    logger = get_logger(__name__)
    
    # Sample instances if dataset is larger than sample_size
    original_count = len(bug_instances)
    if len(bug_instances) > sample_size:
        sampled_instances = random.sample(bug_instances, sample_size)
        is_sampled = True
        logger.info(f"Sampling {sample_size} instances from {original_count} for token calculation")
    else:
        sampled_instances = bug_instances
        is_sampled = False
    
    token_counts = []
    
    for i, bug in enumerate(sampled_instances):
        if i % 100 == 0 and len(sampled_instances) > 100:
            logger.debug(f"Processing token count {i}/{len(sampled_instances)}")
            
        # Count tokens in the bug report
        bug_report_tokens = get_token_count(bug.bug_report, model)
        
        # Count tokens in hints
        hints_tokens = get_token_count(bug.hints_text, model)
        
        # Count tokens in code file list (as it would appear in prompt)
        code_files_text = ', '.join(bug.code_files)
        code_files_tokens = get_token_count(code_files_text, model)
        
        # Total tokens for this instance (approximate prompt size)
        total_tokens = bug_report_tokens + hints_tokens + code_files_tokens
        token_counts.append({
            'instance_id': bug.instance_id,
            'bug_report_tokens': bug_report_tokens,
            'hints_tokens': hints_tokens,
            'code_files_tokens': code_files_tokens,
            'total_tokens': total_tokens,
            'num_code_files': len(bug.code_files)
        })
    
    # Calculate statistics
    total_tokens_list = [tc['total_tokens'] for tc in token_counts]
    
    stats = {
        'total_instances': original_count,  # Report original count
        'sample_size': len(sampled_instances),
        'is_sampled': is_sampled,
        'mean_tokens': sum(total_tokens_list) / len(total_tokens_list),
        'min_tokens': min(total_tokens_list),
        'max_tokens': max(total_tokens_list),
        'total_tokens': sum(total_tokens_list),
        'estimated_total_tokens': int((sum(total_tokens_list) / len(total_tokens_list)) * original_count) if is_sampled else sum(total_tokens_list),
        'detailed_counts': token_counts
    }
    
    return stats

def chunk_code_files(code_files: list[str], max_chunk_tokens: int = 8000, model: str = "gpt-4o") -> list[list[str]]:
    """
    Intelligently chunk code files to fit within token limits.
    
    Args:
        code_files: List of code file paths/content
        max_chunk_tokens: Maximum tokens per chunk (default: 8000 to leave room for prompt)
        model: Model name for token counting
    
    Returns:
        List of chunks, where each chunk is a list of code files
    """
    if not code_files:
        return []
    
    logger = get_logger(__name__)
    chunks = []
    current_chunk = []
    current_chunk_tokens = 0
    
    for i, code_file in enumerate(code_files):
        # Get token count for this individual file
        file_tokens = get_token_count(code_file, model)
        
        # If a single file exceeds max_chunk_tokens, we need to split it further
        if file_tokens > max_chunk_tokens:
            logger.warning(f"File {i} has {file_tokens} tokens, exceeding max chunk size. Will be split separately.")
            
            # Save current chunk if it has content
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                current_chunk_tokens = 0
            
            # Split the large file into smaller pieces
            file_chunks = split_large_file(code_file, max_chunk_tokens, model)
            chunks.extend([[chunk] for chunk in file_chunks])
            continue
        
        # Check if adding this file would exceed the token limit
        if current_chunk_tokens + file_tokens > max_chunk_tokens and current_chunk:
            # Save the current chunk and start a new one
            chunks.append(current_chunk)
            current_chunk = [code_file]
            current_chunk_tokens = file_tokens
        else:
            # Add to current chunk
            current_chunk.append(code_file)
            current_chunk_tokens += file_tokens
    
    # Add the last chunk if it has content
    if current_chunk:
        chunks.append(current_chunk)
    
    logger.info(f"Split {len(code_files)} code files into {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        chunk_tokens = get_token_count("\n\n".join(chunk), model)
        logger.debug(f"Chunk {i+1}: {len(chunk)} files, {chunk_tokens} tokens")
    
    return chunks

def split_large_file(file_content: str, max_tokens: int, model: str = "gpt-4o") -> list[str]:
    """
    Split a large file into smaller chunks while preserving structure.
    
    Args:
        file_content: The content of the file to split
        max_tokens: Maximum tokens per chunk
        model: Model name for token counting
    
    Returns:
        List of file content chunks
    """
    logger = get_logger(__name__)
    
    # Try to split by logical boundaries (functions, classes, etc.)
    lines = file_content.split('\n')
    chunks = []
    current_chunk_lines = []
    current_chunk_tokens = 0
    
    for line in lines:
        line_tokens = get_token_count(line + '\n', model)
        
        # If adding this line would exceed the limit and we have content, save chunk
        if current_chunk_tokens + line_tokens > max_tokens and current_chunk_lines:
            chunks.append('\n'.join(current_chunk_lines))
            current_chunk_lines = [line]
            current_chunk_tokens = line_tokens
        else:
            current_chunk_lines.append(line)
            current_chunk_tokens += line_tokens
    
    # Add the last chunk
    if current_chunk_lines:
        chunks.append('\n'.join(current_chunk_lines))
    
    logger.info(f"Split large file into {len(chunks)} chunks")
    return chunks

def estimate_prompt_tokens(bug_instance, chunk: list[str], model: str = "gpt-4o") -> int:
    """
    Estimate total prompt tokens for a bug instance with a specific code file chunk.
    
    Args:
        bug_instance: BugInstance object
        chunk: List of code files for this chunk
        model: Model name for token counting
    
    Returns:
        Estimated total prompt tokens
    """
    # Calculate base prompt tokens (bug report, hints, etc.)
    base_tokens = (
        get_token_count(bug_instance.bug_report, model) +
        get_token_count(bug_instance.hints_text, model) +
        get_token_count(bug_instance.repo, model) +
        get_token_count(bug_instance.instance_id, model) +
        200  # Approximate tokens for prompt template and formatting
    )
    
    # Add chunk tokens
    chunk_tokens = get_token_count("\n\n".join(chunk), model)
    
    return base_tokens + chunk_tokens