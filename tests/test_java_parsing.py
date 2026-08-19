from method.java_parsing import (
    is_java_path,
    extract_java_symbol_tokens,
    extract_java_skeleton_tokens,
    chunk_java_content,
)

SAMPLE_SOURCE = '''package com.example.shapes;

import java.util.List;
import static java.lang.Math.PI;

/**
 * Utilities for cartesian shape handling.
 */
public class ShapeUtil {

    private static final String NOTE = "not a { real brace or class Fake";
    // class Ignored { this is a comment, not real code }

    public ShapeUtil() {
        System.out.println("init");
    }

    public double computeArea(double radius) {
        if (radius > 0) {
            return PI * radius * radius;
        }
        return 0;
    }

    public abstract void cartesianTransform(double x, double y);
}
'''

INTERFACE_ONLY_SOURCE = '''package com.example.shapes;

public interface Shape {
    double computeArea();
    void cartesianTransform(double x, double y);
}
'''


def test_is_java_path_true_for_dot_java():
    assert is_java_path("com/example/shapes/ShapeUtil.java") is True


def test_is_java_path_false_for_other_extensions():
    assert is_java_path("shape_util.py") is False
    assert is_java_path("ShapeUtil.class") is False


def test_extract_java_symbol_tokens_separates_symbols_from_imports():
    symbol_tokens, import_tokens = extract_java_symbol_tokens(SAMPLE_SOURCE)

    assert "shape" in symbol_tokens and "util" in symbol_tokens  # ShapeUtil (class + constructor)
    assert "compute" in symbol_tokens and "area" in symbol_tokens  # computeArea
    assert "cartesian" in symbol_tokens and "transform" in symbol_tokens  # cartesianTransform

    assert "java" in import_tokens and "util" in import_tokens and "list" in import_tokens
    assert "math" in import_tokens and "pi" in import_tokens
    # Javadoc/comment text should NOT leak into either -- unlike the skeleton variant
    assert "utilities" not in symbol_tokens and "utilities" not in import_tokens


def test_extract_java_symbol_tokens_ignores_comments_and_strings():
    symbol_tokens, _ = extract_java_symbol_tokens(SAMPLE_SOURCE)
    # "class Ignored" only appears inside a line comment; "class Fake" only inside a
    # string literal -- neither should register as a real type declaration.
    assert "ignored" not in symbol_tokens
    assert "fake" not in symbol_tokens
    assert "note" not in symbol_tokens  # a field name, not a class/method declaration


def test_extract_java_symbol_tokens_includes_interface_style_method_without_body():
    # cartesianTransform ends in ';' (no body) -- still a real declaration, still a symbol.
    symbol_tokens, _ = extract_java_symbol_tokens(SAMPLE_SOURCE)
    assert "cartesian" in symbol_tokens and "transform" in symbol_tokens


def test_extract_java_symbol_tokens_returns_empty_pair_for_empty_content():
    assert extract_java_symbol_tokens("") == ([], [])


def test_extract_java_skeleton_tokens_includes_leading_javadoc_and_names():
    tokens = extract_java_skeleton_tokens(SAMPLE_SOURCE)
    assert "cartesian" in tokens  # from the leading Javadoc comment
    assert "shape" in tokens and "util" in tokens
    assert "compute" in tokens and "area" in tokens


def test_extract_java_skeleton_tokens_no_leading_comment_returns_no_comment_tokens():
    no_comment = "package com.example;\n\npublic class Bare {\n    public void run() {}\n}\n"
    tokens = extract_java_skeleton_tokens(no_comment)
    assert "run" in tokens
    assert "bare" in tokens


def test_chunk_java_content_splits_into_header_and_method_chunks():
    chunks = chunk_java_content(SAMPLE_SOURCE)
    # header (package/imports/javadoc/field) + constructor + computeArea + cartesianTransform
    # has no body, so only 2 real method chunks -- 3 chunks total.
    assert len(chunks) == 3
    assert "package com.example.shapes" in chunks[0]
    assert "public ShapeUtil()" in chunks[1]
    assert "public double computeArea" in chunks[2]


def test_chunk_java_content_header_excludes_comment_text():
    # Regression test: the header chunk previously used raw (non-noise-stripped) content,
    # so a leading Javadoc/license comment leaked into it verbatim -- indistinguishable
    # from real code to anything downstream (chunked-embedding ranking, EmbedRank keyword
    # extraction). The header should keep real declarations (package/import/field) but not
    # the comment's own words.
    chunks = chunk_java_content(SAMPLE_SOURCE)
    header = chunks[0]
    assert "cartesian shape handling" not in header
    assert "package com.example.shapes" in header
    assert "import java.util.List" in header
    assert "NOTE" in header


def test_chunk_java_content_handles_nested_braces_in_method_body():
    chunks = chunk_java_content(SAMPLE_SOURCE)
    compute_area_chunk = next(c for c in chunks if "computeArea" in c)
    # The chunk must extend through the nested if-block to the method's own closing
    # brace, not stop early at the inner if-block's brace.
    assert "return PI * radius * radius" in compute_area_chunk
    assert "return 0" in compute_area_chunk
    assert compute_area_chunk.count("{") == compute_area_chunk.count("}")


def test_chunk_java_content_falls_back_to_character_windows_when_no_method_bodies():
    # Every method in an interface ends in ';', not '{' -- no bodies to chunk by.
    chunks = chunk_java_content(INTERFACE_ONLY_SOURCE, max_chunk_chars=60, overlap_chars=10)
    assert len(chunks) > 1
    assert all(len(c) <= 60 for c in chunks)


def test_chunk_java_content_returns_single_window_for_short_content_with_no_methods():
    chunks = chunk_java_content("package com.example;\n", max_chunk_chars=1500, overlap_chars=200)
    assert len(chunks) == 1
