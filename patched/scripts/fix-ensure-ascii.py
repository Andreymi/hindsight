#!/usr/bin/env python3
"""AST-based fix for json.dumps() calls missing ensure_ascii=False.

Unlike semgrep's regex autofix, this correctly handles nested parentheses
(datetime.now(), .to_dict(), tuples) by operating on the AST.

Usage:
    python fix-ensure-ascii.py <file_or_dir> [...]
    python fix-ensure-ascii.py --check <file_or_dir>   # dry-run, exit 1 if fixes needed

Examples:
    python patched/scripts/fix-ensure-ascii.py hindsight-api-slim/hindsight_api/engine/
    python patched/scripts/fix-ensure-ascii.py --check .
"""

import ast
import sys
from pathlib import Path


def find_json_dumps_without_ensure_ascii(source: str) -> list[tuple[int, int, int]]:
    """Find json.dumps() calls missing ensure_ascii=False.

    Returns list of (line, col_offset, end_col_offset) for each fixable call.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    results = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # Match json.dumps(...)
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "dumps":
            if isinstance(func.value, ast.Name) and func.value.id == "json":
                # Check if ensure_ascii is already present
                has_ensure_ascii = any(
                    kw.arg == "ensure_ascii" for kw in node.keywords
                )
                if not has_ensure_ascii:
                    results.append((node.lineno, node.col_offset, node.end_lineno))
    return results


def fix_file(filepath: Path, check_only: bool = False) -> list[tuple[int, str]]:
    """Fix json.dumps() calls in a file. Returns list of (line, preview) for fixes."""
    source = filepath.read_text()
    calls = find_json_dumps_without_ensure_ascii(source)
    if not calls:
        return []

    if check_only:
        return [(line, "") for line, _, _ in calls]

    lines = source.split("\n")
    # Process in reverse order to preserve line numbers
    fixes = []
    for lineno, col_offset, end_lineno in reversed(calls):
        # Find the closing paren of json.dumps() by counting balanced parens
        # Start from the opening paren after "json.dumps"
        start_line = lineno - 1  # 0-indexed

        # Find "json.dumps(" in the start line
        line_text = lines[start_line]
        dumps_pos = line_text.find("json.dumps(", col_offset)
        if dumps_pos == -1:
            continue

        paren_start = dumps_pos + len("json.dumps")
        # Count parens to find the matching close
        depth = 0
        found = False
        for scan_line_idx in range(start_line, min(end_lineno, len(lines))):
            scan_line = lines[scan_line_idx]
            start_col = paren_start if scan_line_idx == start_line else 0
            for col_idx in range(start_col, len(scan_line)):
                ch = scan_line[col_idx]
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        # Found the closing paren — insert ensure_ascii=False before it
                        before = scan_line[:col_idx].rstrip()
                        after = scan_line[col_idx:]
                        # Determine indentation for multi-line
                        if scan_line_idx != start_line:
                            # Multi-line: add on the same line as closing paren
                            indent = len(scan_line) - len(scan_line.lstrip())
                            lines[scan_line_idx] = " " * indent + "ensure_ascii=False,\n" + scan_line
                        else:
                            # Single-line: insert before closing paren
                            if before.endswith(","):
                                lines[scan_line_idx] = before + " ensure_ascii=False" + after
                            else:
                                lines[scan_line_idx] = before + ", ensure_ascii=False" + after
                        fixes.append((lineno, lines[start_line].strip()[:80]))
                        found = True
                        break
                # Skip strings
                elif ch in ('"', "'"):
                    pass  # simplified — AST already validated
            if found:
                break

    if fixes:
        filepath.write_text("\n".join(lines))

    return fixes


def main():
    check_only = "--check" in sys.argv
    paths = [p for p in sys.argv[1:] if p != "--check"]

    if not paths:
        print(__doc__)
        sys.exit(1)

    all_fixes = []
    for path_str in paths:
        path = Path(path_str)
        if path.is_file() and path.suffix == ".py":
            files = [path]
        elif path.is_dir():
            files = sorted(path.rglob("*.py"))
        else:
            continue

        for f in files:
            fixes = fix_file(f, check_only=check_only)
            for line, preview in fixes:
                all_fixes.append((f, line, preview))
                action = "NEEDS FIX" if check_only else "FIXED"
                print(f"  {action}: {f}:{line}")

    if all_fixes:
        print(f"\n{'Would fix' if check_only else 'Fixed'}: {len(all_fixes)} json.dumps() call(s)")
        if check_only:
            sys.exit(1)
    else:
        print("All json.dumps() calls have ensure_ascii=False ✓")


if __name__ == "__main__":
    main()
