"""DAG helper: the book's make_dag_df utility, translated to networkx.

Small causal-model diagrams from an explicit edge list, with a seeded
spring layout for reproducible pictures. Requires the ``dag`` extra
(networkx + matplotlib).
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

__all__ = ["make_dag", "draw_dag"]


def make_dag(
    edges: Sequence[Tuple[str, str]],
    seed: int = 464,
) -> tuple[object, dict[str, object]]:
    """Build a directed graph and a reproducible layout from an edge list.

    Returns ``(graph, pos)`` — a ``networkx.DiGraph`` and its
    ``spring_layout`` positions seeded for reproducibility.
    """
    import networkx as nx

    dag = nx.DiGraph(edges)
    pos = nx.spring_layout(dag, seed=seed)
    return dag, pos


def draw_dag(
    edges: Sequence[Tuple[str, str]],
    seed: int = 464,
    title: Optional[str] = None,
    ax: Optional[Any] = None,
) -> Any:
    """Draw a small DAG with the course's default styling; returns the axes."""
    import matplotlib.pyplot as plt
    import networkx as nx

    dag, pos = make_dag(edges, seed=seed)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    nx.draw_networkx(
        dag, pos=pos, ax=ax, node_size=3000, node_color="#dbe4ff",
        edgecolors="#3b5bdb", linewidths=1.5, font_size=10,
        arrowsize=22, width=1.5,
    )
    if title:
        ax.set_title(title)
    ax.set_axis_off()
    return ax
