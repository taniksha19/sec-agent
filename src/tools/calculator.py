"""Arithmetic for the agent. Parses the expression and walks the syntax tree itself, so only
numbers and + - * / ( ) can run: never eval(), because arguments may echo retrieved text."""

import ast
import math
import operator
import re

from tools.models import CalculateInput, CalculateOutput

OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}
THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d{3}\b)")  # "18,497" -> "18497"


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        return OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
        return OPS[type(node.op)](_eval(node.operand))
    raise ValueError(f"not allowed in a calculation: {type(node).__name__}")


def calculate(inp: CalculateInput) -> CalculateOutput:
    expression = THOUSANDS_RE.sub("", inp.expression).replace("$", "")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"not an arithmetic expression: {inp.expression!r}") from e
    try:
        value = _eval(tree)
    except ZeroDivisionError as e:
        raise ValueError("division by zero") from e
    if not math.isfinite(value):  # e.g. 1e308 * 10; JSON has no inf or NaN
        raise ValueError("result out of range")
    return CalculateOutput(value=value)
