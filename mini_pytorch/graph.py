"""Computation graph inspection without external dependencies."""
from __future__ import annotations

from .tensor import Tensor


def to_dot(output: Tensor) -> str:
    """Return a Graphviz DOT string for the graph ending at ``output``.

    Graphviz itself is optional; the generated .dot text can be pasted into any
    DOT viewer or rendered later with the Graphviz CLI.
    """
    lines = ["digraph Autograd {", "  rankdir=LR;", "  node [shape=record];"]
    visited = set()

    def visit(t: Tensor):
        tid = id(t)
        if tid in visited:
            return
        visited.add(tid)
        label_name = t.name or "Tensor"
        label = f"{label_name}|shape={t.shape}|requires_grad={t.requires_grad}"
        lines.append(f'  t{tid} [label="{{{label}}}"];')
        if t._grad_fn is not None:
            fid = f"f{tid}"
            lines.append(f'  {fid} [shape=box,label="{t.grad_fn}"];')
            lines.append(f"  {fid} -> t{tid};")
            for parent in t._grad_fn.parents:
                visit(parent)
                lines.append(f"  t{id(parent)} -> {fid};")

    visit(output)
    lines.append("}")
    return "\n".join(lines)
