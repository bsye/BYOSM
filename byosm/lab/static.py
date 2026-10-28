from __future__ import annotations

import ast

from .contract import ALLOWED_IMPORTS, BLOCKED_ATTRS, BLOCKED_CALLS, HOOKS

TOP_LEVEL = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.Assign, ast.AnnAssign, ast.Expr)


def _import_problems(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Import):
        roots = [a.name.split(".")[0] for a in node.names]
    elif isinstance(node, ast.ImportFrom):
        roots = [(node.module or "").split(".")[0]] if not node.level else ["relative import"]
    else:
        return []
    return [f"import of {r} is not allowed" for r in roots if r not in ALLOWED_IMPORTS]


def _node_problems(node: ast.AST) -> list[str]:
    out = _import_problems(node)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in BLOCKED_CALLS:
        out.append(f"call to {node.func.id} is not allowed")
    if isinstance(node, ast.Attribute) and (node.attr.startswith("__") or node.attr in BLOCKED_ATTRS):
        out.append(f"attribute {node.attr} is not allowed")
    if isinstance(node, ast.Name) and node.id.startswith("__"):
        out.append(f"name {node.id} is not allowed")
    return out


def _top_level_problems(tree: ast.Module) -> list[str]:
    out = [
        f"line {n.lineno}: only imports, constants and functions may run at import time"
        for n in tree.body
        if not isinstance(n, TOP_LEVEL)
    ]
    out += [
        f"line {n.lineno}: only docstrings may stand alone"
        for n in tree.body
        if isinstance(n, ast.Expr) and not isinstance(n.value, ast.Constant)
    ]
    defined = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    if not defined & set(HOOKS):
        out.append(f"define at least one of {', '.join(HOOKS)}")
    return out


def problems(code: str) -> list[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"line {e.lineno}: {e.msg}"]
    return _top_level_problems(tree) + [p for node in ast.walk(tree) for p in _node_problems(node)]
