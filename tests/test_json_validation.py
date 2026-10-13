import pytest
from fastapi import HTTPException

from json_validation import MAX_JSON_DEPTH, validate_json_depth


def test_nesting_at_limit_is_allowed():
    content = (
            b"[" * MAX_JSON_DEPTH
            + b"0"
            + b"]" * MAX_JSON_DEPTH
    )

    validate_json_depth(content, "Document")


def test_nesting_above_limit_is_rejected():
    depth = MAX_JSON_DEPTH + 1
    content = b"[" * depth + b"0" + b"]" * depth

    with pytest.raises(HTTPException) as error:
        validate_json_depth(content, "Document")

    assert error.value.status_code == 400
    assert error.value.detail == "Document JSON is nested too deeply."


def test_brackets_and_escaped_quotes_in_strings_are_ignored():
    import json

    content = json.dumps({
        "text": 'He said "[[[", then typed {{{ and a backslash \\.',
    }).encode()

    validate_json_depth(content, "Document")