"""Safe arithmetic evaluation for budget maths.

LLMs are unreliable at arithmetic, so the planner delegates every total to this tool.
Expressions are parsed into a Python AST and only numbers, arithmetic operators and a
few whitelisted functions are evaluated -- never `eval()`.
"""

from __future__ import annotations

import ast
import math
import operator
import re
from collections.abc import Callable

MAX_EXPRESSION_LENGTH = 300
MAX_EXPONENT = 100

_BINARY_OPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_FUNCTIONS: dict[str, Callable[..., float]] = {
    "round": round,
    "min": min,
    "max": max,
    "abs": abs,
}
# "1,250.50" -> "1250.50"; only commas used as thousands separators are removed.
_THOUSANDS_SEPARATOR = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")


class CalculatorError(ValueError):
    """Raised for expressions that are invalid or not allowed."""


def evaluate(expression: str) -> float:
    """Evaluate an arithmetic expression such as ``"2 * (18 + 10) + 4 * 3.10"``."""
    cleaned = _THOUSANDS_SEPARATOR.sub("", expression.strip())
    if not cleaned:
        raise CalculatorError("Empty expression.")
    if len(cleaned) > MAX_EXPRESSION_LENGTH:
        raise CalculatorError(f"Expression longer than {MAX_EXPRESSION_LENGTH} characters.")
    try:
        tree = ast.parse(cleaned, mode="eval")
    except SyntaxError as exc:
        raise CalculatorError(f"Could not parse '{expression}'.") from exc
    try:
        result = _eval(tree.body)
    except ZeroDivisionError as exc:
        raise CalculatorError("Division by zero.") from exc
    except (TypeError, OverflowError) as exc:
        raise CalculatorError(f"Could not evaluate '{expression}': {exc}") from exc
    if not isinstance(result, (int, float)) or not math.isfinite(result):
        raise CalculatorError("The result is not a finite real number.")
    return result


def format_number(value: float) -> str:
    """Render whole numbers without decimals and others with up to 4 decimals."""
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _eval(node: ast.expr) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise CalculatorError(f"Exponents above {MAX_EXPONENT} are not allowed.")
        return _BINARY_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval(node.operand))
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCTIONS
        and not node.keywords
    ):
        args = [_eval(arg) for arg in node.args]
        if not args:
            raise CalculatorError(f"{node.func.id}() needs at least one argument.")
        if node.func.id in ("min", "max") and len(args) == 1:
            return args[0]
        if node.func.id == "round" and len(args) == 2:
            return round(args[0], int(args[1]))
        return _FUNCTIONS[node.func.id](*args)
    raise CalculatorError(
        "Only numbers, + - * / // % **, parentheses and round/min/max/abs are allowed."
    )
