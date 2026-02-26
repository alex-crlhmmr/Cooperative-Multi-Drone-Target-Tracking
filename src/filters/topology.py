"""Communication graph utilities for distributed consensus filtering.

Provides adjacency matrix generation, dropout simulation, and visualization
for full, ring, and star topologies with N drones.
"""

import numpy as np


def generate_adjacency(n: int, topology: str) -> np.ndarray:
    """Generate symmetric adjacency matrix for a communication graph.

    Args:
        n: number of drones (nodes)
        topology: "full", "ring", or "star"

    Returns:
        (n, n) binary symmetric matrix (0 on diagonal)
    """
    adj = np.zeros((n, n), dtype=int)

    if topology == "full":
        adj = np.ones((n, n), dtype=int) - np.eye(n, dtype=int)

    elif topology == "ring":
        for i in range(n):
            adj[i, (i + 1) % n] = 1
            adj[i, (i - 1) % n] = 1

    elif topology == "star":
        # Drone 0 is the hub
        for j in range(1, n):
            adj[0, j] = 1
            adj[j, 0] = 1

    else:
        raise ValueError(f"Unknown topology: {topology}. Use 'full', 'ring', or 'star'.")

    return adj


def apply_dropout(adj: np.ndarray, prob: float, rng: np.random.Generator) -> np.ndarray:
    """Apply symmetric random dropout to adjacency matrix.

    Each edge independently fails with probability `prob`.
    Dropout is symmetric: if (i,j) drops, (j,i) drops too.

    Args:
        adj: (n, n) adjacency matrix
        prob: dropout probability per edge [0, 1]
        rng: numpy random generator

    Returns:
        (n, n) masked adjacency matrix
    """
    if prob <= 0.0:
        return adj.copy()
    if prob >= 1.0:
        return np.zeros_like(adj)

    n = adj.shape[0]
    # Generate upper-triangle mask, mirror to lower
    mask = np.ones((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] and rng.random() < prob:
                mask[i, j] = 0
                mask[j, i] = 0

    return adj * mask


def draw_topology(adj: np.ndarray, title: str = "", save_path: str | None = None):
    """Draw communication graph using NetworkX + matplotlib.

    Args:
        adj: (n, n) adjacency matrix
        title: plot title
        save_path: if provided, save figure to this path
    """
    import matplotlib.pyplot as plt
    try:
        import networkx as nx
    except ImportError:
        print("NetworkX not installed — skipping topology plot")
        return

    G = nx.from_numpy_array(adj)
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    pos = nx.circular_layout(G)
    nx.draw(
        G, pos, ax=ax,
        with_labels=True,
        node_color="#2196F3",
        node_size=500,
        font_size=12,
        font_color="white",
        font_weight="bold",
        edge_color="#666666",
        width=2.0,
    )
    ax.set_title(title or "Communication Graph", fontsize=13)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()
