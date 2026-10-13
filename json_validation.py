from fastapi import HTTPException


MAX_JSON_DEPTH = 64


def validate_json_depth(content: bytes, label: str) -> None:
    depth = 0
    in_string = False
    escaped = False

    for character in content:
        if in_string:
            if escaped:
                escaped = False
            elif character == 92:  # Backslash
                escaped = True
            elif character == 34:  # Double quote
                in_string = False
            continue

        if character == 34:
            in_string = True
        elif character in (91, 123):  # [ or {
            depth += 1
            if depth > MAX_JSON_DEPTH:
                raise HTTPException(
                    status_code=400,
                    detail=f"{label} JSON is nested too deeply.",
                )
        elif character in (93, 125):  # ] or }
            depth -= 1