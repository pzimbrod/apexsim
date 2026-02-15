"""AST-level validation for repository-wide docstring contracts."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCAN_ROOTS = ("src", "examples", "tests", "scripts")
IGNORED_PARAM_NAMES = {"self", "cls"}


def _has_section(docstring: str, section: str) -> bool:
    """Check whether a docstring contains an exact section header line.

    Args:
        docstring: Full docstring text.
        section: Section name without trailing colon.

    Returns:
        ``True`` if the section header exists.
    """
    target = f"{section}:"
    return any(line.strip() == target for line in docstring.splitlines())


def _has_non_self_params(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Detect whether a callable has parameters other than ``self``/``cls``.

    Args:
        node: Function or method node to inspect.

    Returns:
        ``True`` when at least one relevant parameter is present.
    """
    all_args: list[ast.arg] = []
    all_args.extend(node.args.posonlyargs)
    all_args.extend(node.args.args)
    all_args.extend(node.args.kwonlyargs)
    if node.args.vararg is not None:
        all_args.append(node.args.vararg)
    if node.args.kwarg is not None:
        all_args.append(node.args.kwarg)
    return any(arg.arg not in IGNORED_PARAM_NAMES for arg in all_args)


def _returns_non_none(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Determine whether a callable annotation indicates a non-``None`` return.

    Args:
        node: Function or method node to inspect.

    Returns:
        ``True`` if the return annotation is present and not ``None``.
    """
    if node.returns is None:
        return False
    if isinstance(node.returns, ast.Name) and node.returns.id == "None":
        return False
    return not (isinstance(node.returns, ast.Constant) and node.returns.value is None)


class DocstringContractTests(unittest.TestCase):
    """Validate docstring presence and section contracts for public code."""

    def test_docstring_contracts(self) -> None:
        """Enforce summary/Args/Returns coverage based on callable signatures."""
        violations: list[str] = []

        for root_name in SCAN_ROOTS:
            for path in sorted((REPO_ROOT / root_name).rglob("*.py")):
                tree = ast.parse(path.read_text(encoding="utf-8"))
                relative_path = path.relative_to(REPO_ROOT)
                self._collect_violations(tree, relative_path, violations)

        if violations:
            formatted = "\n".join(f"- {item}" for item in violations)
            self.fail(f"Docstring contract violations:\n{formatted}")

    def _collect_violations(
        self,
        tree: ast.Module,
        path: Path,
        violations: list[str],
    ) -> None:
        """Collect docstring contract violations from one module AST.

        Args:
            tree: Parsed module AST.
            path: Repository-relative path to the module.
            violations: Mutable collection for discovered violations.
        """
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                self._check_class(path, node, violations)
                continue
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._check_callable(path, node, violations, owner=None)

    def _check_class(self, path: Path, node: ast.ClassDef, violations: list[str]) -> None:
        """Validate class-level and method-level docstring contracts.

        Args:
            path: Repository-relative path to the module.
            node: Class node to inspect.
            violations: Mutable collection for discovered violations.
        """
        class_doc = ast.get_docstring(node)
        if class_doc is None:
            violations.append(f"{path}:{node.lineno} missing class docstring for `{node.name}`")

        for member in node.body:
            if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._check_callable(path, member, violations, owner=node.name)

    def _check_callable(
        self,
        path: Path,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        violations: list[str],
        owner: str | None,
    ) -> None:
        """Validate callable docstring summary and sections.

        Args:
            path: Repository-relative path to the module.
            node: Function or method node to inspect.
            violations: Mutable collection for discovered violations.
            owner: Optional owning class name for methods.
        """
        qualified_name = f"{owner}.{node.name}" if owner is not None else node.name
        doc = ast.get_docstring(node)
        if doc is None:
            violations.append(f"{path}:{node.lineno} missing docstring for `{qualified_name}`")
            return

        if _has_non_self_params(node) and not _has_section(doc, "Args"):
            violations.append(f"{path}:{node.lineno} missing Args for `{qualified_name}`")

        if _returns_non_none(node) and not _has_section(doc, "Returns"):
            violations.append(f"{path}:{node.lineno} missing Returns for `{qualified_name}`")
