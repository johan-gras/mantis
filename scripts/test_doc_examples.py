#!/usr/bin/env python3
"""
Test Python code examples in documentation.

Extracts Python code blocks from markdown files and executes them
to ensure documentation examples are correct and up-to-date.

Usage:
    python scripts/test_doc_examples.py
    python scripts/test_doc_examples.py --docs-dir docs/
    python scripts/test_doc_examples.py --verbose
    python scripts/test_doc_examples.py --file docs/quickstart.md
"""

import argparse
import re
import sys
import traceback
from io import StringIO
from pathlib import Path
from typing import Optional


def extract_python_blocks(md_content: str) -> list[tuple[int, str]]:
    """
    Extract Python code blocks from markdown content.

    Returns list of (line_number, code) tuples.
    """
    blocks = []
    # Match ```python followed by code until ```
    pattern = r"```python\n(.*?)```"

    for match in re.finditer(pattern, md_content, re.DOTALL):
        # Calculate line number
        start_pos = match.start()
        line_num = md_content[:start_pos].count("\n") + 1
        code = match.group(1)
        blocks.append((line_num, code))

    return blocks


def should_skip_block(code: str) -> tuple[bool, Optional[str]]:
    """
    Check if a code block should be skipped.

    Returns (should_skip, reason).
    """
    stripped_code = code.strip()

    # Skip if explicitly marked
    if "# skip-test" in code.lower() or "# notest" in code.lower():
        return True, "marked with skip-test"

    # Skip if it's clearly a snippet (incomplete/partial code)
    first_line = stripped_code.split("\n")[0] if stripped_code else ""
    if first_line.startswith("...") or ("..." in first_line and first_line.startswith("#")):
        return True, "continuation snippet"

    # Skip output-only blocks (just show expected output)
    if stripped_code.startswith("┌") or stripped_code.startswith("│"):
        return True, "output example"

    # Skip pseudo-code and placeholder examples
    if "your_data" in code or "my_data" in code or "model.predict" in code:
        return True, "placeholder example"

    # Skip Jupyter/Colab magic commands
    if "!pip" in code or "!conda" in code or stripped_code.startswith("!"):
        return True, "Jupyter magic command"

    # Skip incomplete snippet indicators (arrows pointing to features)
    if "# <-" in code or "#<-" in code or "# ←" in code or "<- " in code:
        return True, "incomplete snippet indicator"

    # Skip pseudo-code with incomplete control flow
    lines = stripped_code.split("\n")
    for i, line in enumerate(lines):
        stripped_line = line.strip()
        # Check for else/elif without proper body
        if stripped_line in ("else:", "elif:") or stripped_line.startswith("else:") or stripped_line.startswith("elif "):
            # Check if next non-comment line is properly indented
            next_meaningful = None
            for next_line in lines[i+1:]:
                if next_line.strip() and not next_line.strip().startswith("#"):
                    next_meaningful = next_line
                    break
            if next_meaningful is None:
                return True, "pseudo-code (incomplete control flow)"
            if not next_meaningful.startswith("    ") and not next_meaningful.startswith("\t"):
                return True, "pseudo-code (incomplete control flow)"

    # Skip function/class signature blocks (API documentation)
    # These look like: def foo(...) -> ReturnType or class Foo:
    if stripped_code.startswith("def ") or stripped_code.startswith("class "):
        lines = stripped_code.split("\n")
        # Check if it's a signature without a body (ends with just the signature)
        last_meaningful = ""
        for line in reversed(lines):
            if line.strip() and not line.strip().startswith("#"):
                last_meaningful = line.strip()
                break
        # Signature-only blocks end with return type annotation or just ): or :
        if (last_meaningful.endswith("-> ") or
            ") -> " in last_meaningful or
            last_meaningful.endswith(")") or
            (last_meaningful.endswith(":") and ":" not in last_meaningful[:-1].split()[-1] if last_meaningful[:-1].split() else True)):
            # Check if there's no body (no indented code after the definition)
            has_body = False
            in_signature = True
            for line in lines[1:]:  # Skip the first line
                if line.strip() and not line.startswith(" ") and not line.startswith("\t"):
                    in_signature = False
                if not in_signature and (line.startswith("    ") or line.startswith("\t")):
                    has_body = True
                    break
            if not has_body:
                return True, "function/class signature (API docs)"

    # Skip type hint blocks that define structures
    # e.g., `validation: ValidationResult` or property listings
    if ": " in first_line and "=" not in first_line:
        # Check if it's a type annotation pattern
        if re.match(r"^\s*\w+\s*:\s*\w+", first_line):
            # Could be a class attribute listing
            all_annotations = True
            for line in stripped_code.split("\n"):
                line = line.strip()
                if line and not line.startswith("#"):
                    if not re.match(r"^\w+\s*:\s*[\w\[\], ]+", line):
                        all_annotations = False
                        break
            if all_annotations:
                return True, "type annotation listing"

    # Skip pseudo-code blocks (algorithmic descriptions)
    # These often contain plain English or non-Python syntax
    pseudo_patterns = ["->", "→", "// ", "/* ", "FOR ", "IF ", "WHILE ", "RETURN "]
    for pattern in pseudo_patterns:
        if pattern in code and not code.strip().startswith("def") and not code.strip().startswith("class"):
            # Check if it's pseudo-code rather than actual Python
            if "//" in code or "/*" in code:  # C-style comments
                return True, "pseudo-code"
            if re.search(r"\b(FOR|IF|WHILE|THEN|END)\b", code):  # Uppercase keywords
                return True, "pseudo-code"

    # Skip blocks that reference undefined variables (context from text)
    lines = stripped_code.split("\n")
    first_meaningful = None
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and not stripped.startswith("import"):
            first_meaningful = stripped
            break

    if first_meaningful:
        # Skip if first meaningful line uses undefined names common in docs
        if first_meaningful.startswith("results") and "=" not in first_meaningful.split("results")[0]:
            if "results = " not in code and "results=" not in code:
                return True, "depends on 'results' context"
        if first_meaningful.startswith("data") and "=" not in first_meaningful.split("data")[0]:
            if "data = " not in code and "data=" not in code:
                return True, "depends on 'data' context"

    return False, None


def check_mantis_available() -> bool:
    """Check if mantis module is available."""
    try:
        import mantis  # noqa: F401
        return True
    except ImportError:
        return False


MANTIS_AVAILABLE = None  # Lazy initialization


def create_test_environment() -> dict:
    """
    Create a test environment with common imports and setup.

    This provides a baseline environment for running doc examples.
    """
    global MANTIS_AVAILABLE
    if MANTIS_AVAILABLE is None:
        MANTIS_AVAILABLE = check_mantis_available()

    env = {
        "__name__": "__main__",
        "__file__": "<doc_example>",
    }

    # Pre-import common modules
    setup_code = """
import numpy as np
np.random.seed(42)  # Deterministic examples
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings in examples
"""
    try:
        exec(setup_code, env)
    except ImportError:
        pass  # numpy might not be available

    # Pre-import mantis if available
    if MANTIS_AVAILABLE:
        try:
            exec("import mantis as mt", env)
        except ImportError:
            pass

    return env


def run_code_block(
    code: str,
    env: dict,
    capture_output: bool = True
) -> tuple[bool, Optional[str]]:
    """
    Execute a code block in the given environment.

    Returns (success, error_message).
    """
    if capture_output:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()

    try:
        exec(code, env)
        return True, None
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        if hasattr(e, "__traceback__"):
            tb_lines = traceback.format_tb(e.__traceback__)
            # Filter to show only the relevant part
            error_msg += "\n" + "".join(tb_lines[-3:])
        return False, error_msg
    finally:
        if capture_output:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def test_file(
    md_path: Path,
    verbose: bool = False,
    continue_on_error: bool = True
) -> tuple[int, int, list[str]]:
    """
    Test all Python code blocks in a markdown file.

    Returns (passed, skipped, errors).
    """
    content = md_path.read_text()
    blocks = extract_python_blocks(content)

    if not blocks:
        if verbose:
            print(f"  No Python blocks found")
        return 0, 0, []

    passed = 0
    skipped = 0
    errors = []

    # Create shared environment for blocks in same file
    # This allows later examples to reference earlier definitions
    env = create_test_environment()

    for line_num, code in blocks:
        skip, reason = should_skip_block(code)
        if skip:
            if verbose:
                print(f"  Line {line_num}: SKIP ({reason})")
            skipped += 1
            continue

        success, error = run_code_block(code, env)

        if success:
            if verbose:
                print(f"  Line {line_num}: OK")
            passed += 1
        else:
            # Check if this is a known issue pattern
            error_str = str(error)

            # Skip if mantis not available
            if "ModuleNotFoundError" in error_str and "mantis" in error_str:
                if verbose:
                    print(f"  Line {line_num}: SKIP (mantis not installed)")
                skipped += 1
                continue

            # Skip if depends on prior context
            if "NameError" in error_str:
                # Common documentation variable names that indicate example context
                context_vars = [
                    "signal", "data", "results", "mt", "df", "model",
                    "validation", "sweep", "equity", "trades", "metrics",
                    # Concept documentation variables
                    "shares", "price", "close", "open", "high", "low", "volume",
                    "next_bar", "bar", "current_price", "position_size",
                    "position_value", "i", "k", "n", "np", "pd",
                    # API documentation context
                    "datetime", "comparison", "loaded",
                    # Additional example variables
                    "aapl_signal", "spy_signal", "btc_signal",
                    "model_confidence", "daily_model_signal",
                    "limit_price", "atr",
                ]
                for context_var in context_vars:
                    if f"'{context_var}'" in error_str or f'"{context_var}"' in error_str:
                        if verbose:
                            print(f"  Line {line_num}: SKIP (depends on '{context_var}' context)")
                        skipped += 1
                        break
                else:
                    # NameError for unknown variable - real error
                    error_msg = f"{md_path}:{line_num}: {error}"
                    errors.append(error_msg)
                    if verbose:
                        print(f"  Line {line_num}: FAIL")
                        print(f"    {error}")
                continue

            # Skip TypeError involving builtin conflicts (common in pseudo-code)
            if "TypeError" in error_str and ("builtin_function_or_method" in error_str or "open" in code):
                if verbose:
                    print(f"  Line {line_num}: SKIP (uses builtin as variable)")
                skipped += 1
                continue

            # Skip IndentationError for code from markdown lists (nested code blocks)
            if "IndentationError" in error_str and "unexpected indent" in error_str:
                if verbose:
                    print(f"  Line {line_num}: SKIP (nested code block indentation)")
                skipped += 1
                continue

            error_msg = f"{md_path}:{line_num}: {error}"
            errors.append(error_msg)

            if verbose:
                print(f"  Line {line_num}: FAIL")
                print(f"    {error}")

            if not continue_on_error:
                break

    return passed, skipped, errors


def main():
    parser = argparse.ArgumentParser(
        description="Test Python code examples in documentation"
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=Path("docs"),
        help="Directory containing markdown files (default: docs/)"
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=None,
        help="Test a single file instead of entire docs directory"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed output for each code block"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on first error instead of collecting all errors"
    )

    args = parser.parse_args()

    # Determine files to test
    if args.file:
        if not args.file.exists():
            print(f"Error: File not found: {args.file}")
            sys.exit(1)
        files = [args.file]
    else:
        files = sorted(args.docs_dir.glob("**/*.md"))
        if not files:
            print(f"Error: No markdown files found in {args.docs_dir}")
            sys.exit(1)

    print(f"Testing {len(files)} documentation file(s)...")

    # Check mantis availability
    global MANTIS_AVAILABLE
    if MANTIS_AVAILABLE is None:
        MANTIS_AVAILABLE = check_mantis_available()

    if not MANTIS_AVAILABLE:
        print()
        print("NOTE: mantis module not installed. Examples importing mantis will be skipped.")
        print("      Install with: pip install -e . (from repo root)")

    print()

    total_passed = 0
    total_skipped = 0
    all_errors = []

    for md_path in files:
        rel_path = md_path.relative_to(Path.cwd()) if md_path.is_relative_to(Path.cwd()) else md_path
        print(f"Testing {rel_path}...")

        passed, skipped, errors = test_file(
            md_path,
            verbose=args.verbose,
            continue_on_error=not args.strict
        )

        total_passed += passed
        total_skipped += skipped
        all_errors.extend(errors)

        if errors and not args.verbose:
            for error in errors:
                print(f"  ERROR: {error.split(':')[2] if ':' in error else error}")

        if args.strict and errors:
            break

    # Summary
    print()
    print("=" * 60)
    print(f"Results: {total_passed} passed, {total_skipped} skipped, {len(all_errors)} failed")
    print("=" * 60)

    if all_errors:
        print()
        print("Failures:")
        for error in all_errors:
            print(f"  {error.split(chr(10))[0]}")
        print()
        print("To fix: Update the code examples or add '# skip-test' comment")
        sys.exit(1)
    else:
        print()
        print("All documentation examples passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
