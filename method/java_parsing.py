"""Java-source counterpart to the ast.parse() paths in bm25_retriever.py and
embedding_retriever.py -- Python's ast module doesn't parse Java, so those two files
silently fell back to weaker representations (path-only BM25 tokens, coarse fixed-window
embedding chunks) for every Java file, which is why all 4 BM25 representations tied
exactly and the embedding chunker ran unusually fast on Bench4BL (see
docs/bench4bl_result.md, "Java support"). This is a regex/lexical scanner, not a real
grammar-based parser (a tree-sitter-java version was tried and works, but was dropped to
avoid a new pip dependency that would need offline installation on MN5, which has no
internet -- same class of friction as past accelerate/psutil MN5 blockers). It doesn't
build a parse tree; it strips comments/strings so keywords/braces inside them can't cause
false matches, then finds declarations via regex and method bodies via brace-depth
counting. Good enough for typical, conventionally-formatted Java (which Bench4BL's Apache
Commons/Spring/Wildfly projects are) -- not a substitute for real parsing on adversarial
or unusually-formatted input.
"""

import re

_BLOCK_COMMENT_RE = re.compile(r'/\*.*?\*/', re.DOTALL)
_LINE_COMMENT_RE = re.compile(r'//[^\n]*')
_STRING_RE = re.compile(r'"(?:\\.|[^"\\])*"')
_CHAR_RE = re.compile(r"'(?:\\.|[^'\\])*'")

_TYPE_DECL_RE = re.compile(r'\b(?:class|interface|enum|record)\s+(\w+)')
_IMPORT_RE = re.compile(r'^\s*import\s+(?:static\s+)?([\w.]+(?:\.\*)?)\s*;', re.MULTILINE)

# Heuristic for both regular methods and constructors: a modifier keyword, then any
# run of non-terminator characters (return type, generics, annotations -- deliberately
# unconstrained), then NAME(...), then an optional throws clause, then '{' (has a body)
# or ';' (interface/abstract method, no body). Excluding ';', '{', '}', '=' from the
# middle group is what keeps a match from leaking across statement/block/assignment
# boundaries into unrelated code -- e.g. a field initializer's own method call, or a
# later, unrelated method several lines down.
_METHOD_SIG_RE = re.compile(
    r'\b(?:public|protected|private|static|final|synchronized|abstract|native|default)\b'
    r'[^;{}=]*?\b(\w+)\s*\([^;{}]*\)\s*(?:throws\s+[\w.,\s]+)?\s*([{;])'
)


def is_java_path(path: str) -> bool:
    return path.endswith(".java")


def _strip_noise(content: str) -> str:
    """Blank out block/line comments and string/char literals with same-length
    whitespace (newlines preserved), so regex signature matching and brace-depth
    chunking below can't be confused by a keyword or brace that happens to appear
    inside a comment or a string literal."""
    def blank(m):
        text = m.group(0)
        return "".join(c if c == "\n" else " " for c in text)

    cleaned = _BLOCK_COMMENT_RE.sub(blank, content)
    cleaned = _LINE_COMMENT_RE.sub(blank, cleaned)
    cleaned = _STRING_RE.sub(blank, cleaned)
    cleaned = _CHAR_RE.sub(blank, cleaned)
    return cleaned


def _leading_comment(content: str) -> str | None:
    """Best-effort Javadoc/header comment before the first real code -- Java's rough
    docstring equivalent. None if the file starts with code, not a comment."""
    for m in re.finditer(r'/\*.*?\*/|//[^\n]*|\S', content, re.DOTALL):
        text = m.group(0)
        return text if text.startswith(("/*", "//")) else None
    return None


def _method_signatures(cleaned: str) -> list[tuple[str, int, int]]:
    """Returns (name, match_start, terminator_index) for every method/constructor
    signature found in the (already noise-stripped) content. terminator_index points at
    the '{' or ';' that ends the signature -- '{' means chunk_java_content() can extract
    a real body via brace counting, ';' means an interface/abstract declaration with no
    body (name still usable as a symbol token, just nothing to chunk)."""
    return [(m.group(1), m.start(), m.end() - 1) for m in _METHOD_SIG_RE.finditer(cleaned)]


def _find_matching_brace(cleaned: str, open_idx: int) -> int:
    """Index of the '}' that closes the '{' at open_idx, via simple depth counting on
    the noise-stripped text (so braces inside strings/comments can't throw off the
    count). Falls back to end-of-content if the file is malformed enough to never
    balance -- shouldn't happen on real compiled-at-some-point source."""
    depth = 0
    for i in range(open_idx, len(cleaned)):
        if cleaned[i] == '{':
            depth += 1
        elif cleaned[i] == '}':
            depth -= 1
            if depth == 0:
                return i
    return len(cleaned) - 1


def extract_java_symbol_tokens(content: str) -> tuple[list[str], list[str]]:
    """Java counterpart to bm25_retriever._extract_symbol_tokens(): class/interface/enum/
    record/method/constructor names as symbol_tokens, imported package/class names as
    import_tokens."""
    from method.bm25_retriever import _tokenize_path  # local import avoids a circular import

    cleaned = _strip_noise(content)

    symbol_tokens = []
    for m in _TYPE_DECL_RE.finditer(cleaned):
        symbol_tokens += _tokenize_path(m.group(1))
    for name, _, _ in _method_signatures(cleaned):
        symbol_tokens += _tokenize_path(name)

    import_tokens = []
    for m in _IMPORT_RE.finditer(cleaned):
        text = m.group(1).rstrip('*').rstrip('.')
        import_tokens += _tokenize_path(text)

    return symbol_tokens, import_tokens


def extract_java_skeleton_tokens(content: str) -> list[str]:
    """Java counterpart to bm25_retriever._extract_skeleton_tokens(): the leading
    Javadoc/comment plus class/interface/enum/record/method/constructor names."""
    from method.bm25_retriever import _tokenize_path, _tokenize_query

    tokens = []
    comment = _leading_comment(content)
    if comment:
        tokens += _tokenize_query(comment[:300])

    cleaned = _strip_noise(content)
    for m in _TYPE_DECL_RE.finditer(cleaned):
        tokens += _tokenize_path(m.group(1))
    for name, _, _ in _method_signatures(cleaned):
        tokens += _tokenize_path(name)

    return tokens


def chunk_java_content(content: str, max_chunk_chars: int = 1500, overlap_chars: int = 200) -> list[str]:
    """Java counterpart to embedding_retriever._chunk_file_content(). Deliberately chunks
    at METHOD granularity rather than mirroring the Python path's top-level-declaration
    granularity: Java's one-public-class-per-file convention means chunking only at the
    top level would collapse most files right back to one whole-class chunk -- the exact
    coarse-fallback problem this module exists to fix (see docs/bench4bl_result.md,
    "Java support"). One header chunk (package/imports/leading comment) plus one chunk
    per method/constructor body found via brace-depth counting from each signature's '{'.
    Falls back to fixed-size overlapping windows if no method body was found (e.g. an
    interface with no default methods, or anything the regex heuristic misses).
    """
    cleaned = _strip_noise(content)
    sigs = sorted(_method_signatures(cleaned), key=lambda t: t[1])

    method_chunks = []
    first_start = None
    for _, start, term_idx in sigs:
        if cleaned[term_idx] != '{':
            continue  # interface/abstract declaration, no body to chunk
        end_idx = _find_matching_brace(cleaned, term_idx)
        if first_start is None:
            first_start = start
        method_chunks.append(content[start:end_idx + 1])

    if method_chunks:
        chunks = []
        # noise-stripped, not raw content -- a raw header slice includes the leading
        # Javadoc/license comment verbatim (e.g. the full Apache License boilerplate),
        # which then gets embedded as if it were real code content, indistinguishable
        # from a method body to anything downstream (chunked-embedding ranking, EmbedRank
        # keyword extraction -- both saw this leak into results). cleaned[:first_start]
        # keeps the real signal (package/import declarations) and blanks the comment body.
        header = cleaned[:first_start].strip()
        if header:
            chunks.append(header)
        chunks += method_chunks
        return chunks

    step = max(max_chunk_chars - overlap_chars, 1)
    return [content[i:i + max_chunk_chars] for i in range(0, len(content), step)]
