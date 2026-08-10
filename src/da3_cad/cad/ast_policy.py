"""Conservative AST allow-list for non-adversarial generated CadQuery code."""

from __future__ import annotations

import ast
from typing import Final

ALLOWED_IMPORTS: Final = {"cadquery"}
ALLOWED_BUILTINS: Final = {
    "abs",
    "dict",
    "float",
    "int",
    "len",
    "list",
    "max",
    "min",
    "range",
    "tuple",
}
FORBIDDEN_NAMES: Final = {
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "globals",
    "help",
    "input",
    "locals",
    "open",
    "quit",
    "setattr",
    "getattr",
    "vars",
    "__builtins__",
}


class AstPolicyError(ValueError):
    """Raised when generated code exceeds the accepted CadQuery subset."""


class _PolicyVisitor(ast.NodeVisitor):
    allowed_nodes: Final = (
        ast.Module,
        ast.Import,
        ast.alias,
        ast.Assign,
        ast.AnnAssign,
        ast.Expr,
        ast.Name,
        ast.Load,
        ast.Store,
        ast.Constant,
        ast.Dict,
        ast.List,
        ast.Tuple,
        ast.Subscript,
        ast.Slice,
        ast.Attribute,
        ast.Call,
        ast.keyword,
        ast.BinOp,
        ast.UnaryOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.USub,
        ast.UAdd,
    )

    def generic_visit(self, node: ast.AST) -> None:
        if not isinstance(node, self.allowed_nodes):
            raise AstPolicyError(f"AST node is not allowed: {type(node).__name__}")
        super().generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name not in ALLOWED_IMPORTS:
                raise AstPolicyError(f"import is not allowed: {alias.name}")
            if alias.asname not in {None, "cq"}:
                raise AstPolicyError("cadquery may only be imported as 'cq'")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in FORBIDDEN_NAMES or node.id.startswith("__"):
            raise AstPolicyError(f"name is not allowed: {node.id}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("_"):
            raise AstPolicyError(f"private/dunder attribute is not allowed: {node.attr}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            if node.func.id not in ALLOWED_BUILTINS:
                raise AstPolicyError(f"function call is not allowed: {node.func.id}")
        elif not isinstance(node.func, ast.Attribute):
            raise AstPolicyError("only CadQuery methods and selected builtins may be called")
        self.generic_visit(node)


def validate_source(source: str) -> ast.Module:
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as error:
        raise AstPolicyError(f"invalid Python syntax: {error}") from error
    _PolicyVisitor().visit(tree)
    assigned = {
        target.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    if "r" not in assigned:
        raise AstPolicyError("program must assign the final CadQuery object to 'r'")
    return tree
