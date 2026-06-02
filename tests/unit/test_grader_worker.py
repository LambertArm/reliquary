"""Tests for the sandbox worker's call-only eval logic."""


def test_call_function_returns_primitive_output():
    from reliquary.environment.grader.worker import evaluate_call

    output, status = evaluate_call(
        "def add(a, b): return a + b",
        {"kind": "function", "name": "add"},
        [1, 2],
        {},
        timeout_s=5.0,
    )
    assert status == "ok"
    assert output == 3


def test_call_method_entrypoint():
    from reliquary.environment.grader.worker import evaluate_call

    code = "class Solution:\n    def inc(self, x): return x + 1"
    output, status = evaluate_call(
        code,
        {"kind": "method", "class_name": "Solution", "method": "inc"},
        [4],
        {},
        timeout_s=5.0,
    )
    assert status == "ok"
    assert output == 5


def test_import_math_allowed():
    from reliquary.environment.grader.worker import evaluate_call

    code = "import math\ndef f(x): return math.sqrt(x)"
    output, status = evaluate_call(
        code,
        {"kind": "function", "name": "f"},
        [9],
        {},
        timeout_s=5.0,
    )
    assert status == "ok"
    assert output == 3.0


def test_forbidden_import_is_reported():
    from reliquary.environment.grader.worker import evaluate_call

    output, status = evaluate_call(
        "import os\ndef f(): return True",
        {"kind": "function", "name": "f"},
        [],
        {},
        timeout_s=5.0,
    )
    assert status == "forbidden_import"
    assert output is None


def test_custom_object_output_is_rejected():
    from reliquary.environment.grader.worker import evaluate_call

    code = """
class AlwaysEqual:
    def __eq__(self, other): return True
def f():
    return AlwaysEqual()
"""
    output, status = evaluate_call(
        code,
        {"kind": "function", "name": "f"},
        [],
        {},
        timeout_s=5.0,
    )
    assert status == "bad_output"
    assert output is None


def test_print_does_not_leak_to_stdout(capsys):
    from reliquary.environment.grader.worker import evaluate_call

    output, status = evaluate_call(
        'print("malicious noise")\ndef f(): return 7',
        {"kind": "function", "name": "f"},
        [],
        {},
        timeout_s=5.0,
    )
    assert status == "ok"
    assert output == 7
    captured = capsys.readouterr()
    assert "malicious noise" not in captured.out
    assert "malicious noise" not in captured.err


def test_runtime_exception_is_not_successful_none():
    from reliquary.environment.grader.worker import evaluate_call

    output, status = evaluate_call(
        "def f():\n    raise RuntimeError('boom')",
        {"kind": "function", "name": "f"},
        [],
        {},
        timeout_s=5.0,
    )
    assert status == "runtime_error"
    assert output is None


def test_compile_tamper_fails_without_passing():
    from reliquary.environment.grader.worker import evaluate_call

    code = """
import builtins
def f(): return 1
"""
    output, status = evaluate_call(
        code,
        {"kind": "function", "name": "f"},
        [],
        {},
        timeout_s=5.0,
    )
    assert status == "forbidden_import"
    assert output is None
