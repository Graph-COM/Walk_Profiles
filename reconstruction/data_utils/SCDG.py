import numpy as np
import networkx as nx
from scipy.spatial import KDTree
from scipy.stats import lognorm
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt




def spatially_coherent_directed_graph(n, d, k, degreedist=None):
    """
    Generate a spatially coherent directed graph.

    Parameters:
        n (int): Number of vertices.
        d (int): Dimension of space (e.g., 2 for 2D).
        k (int): Number of clusters.
        degreedist (scipy.stats distribution, optional): Degree distribution.

    Returns:
        G (networkx.DiGraph): Directed graph.
        X (np.ndarray): Node positions.
        groups (np.ndarray): Cluster assignments.
    """
    # Generate random points in [0,1]^d space
    X = np.random.uniform(0, 1, size=(n, d))

    # Assign degrees based on the given distribution
    if degreedist is None:
        degreedist = lognorm(s=1, scale=np.exp(4))  # Log-normal default
    degs = degreedist.rvs(size=n).astype(int)
    for i in range(len(degs)):
        if degs[i] >= n-1:
            degs[i] = n-1

    # Cluster points into `k` groups using KMeans with a fixed seed
    kmeans = KMeans(n_clusters=k, init="k-means++", random_state=SEED)
    groups = kmeans.fit_predict(X)

    # Generate a random direction vector for each cluster
    Y = np.random.normal(size=(k, d))  # Cluster direction vectors

    # Compute directional edges
    ei, ej = directional_edges(X, Y, degs, groups)

    # Create a directed graph
    G = nx.DiGraph()
    G.add_edges_from(zip(ei, ej))

    return G, X, groups


def _orient_edge(xi, xj, i, j, g):
    """
    Orient the edge between points `xi` and `xj` based on direction vector `g`.

    Returns:
        (i, j) if aligned with `g`, else (j, i)
    """
    return (j, i) if np.dot(xj - xi, g) > 0 else (i, j)


def directional_edges(X, Y, degs, groups):
    """
    Generate edges based on nearest neighbors and directional vectors.

    Returns:
        ei (list): Source nodes of directed edges.
        ej (list): Target nodes of directed edges.
    """
    tree = KDTree(X)  # Nearest neighbor search
    ei, ej = [], []

    for i, (x, deg, group) in enumerate(zip(X, degs, groups)):
        _, idxs = tree.query(x, k=int(deg) + 1)  # Find `deg` nearest neighbors
        # if deg >= nodes, this could produce multi-edges. Drop it for now
        idxs = np.array(idxs)
        if idxs.shape == ():
            continue
        idxs = np.array(list(set(idxs)))

        for j in idxs:
            if i != j:  # Avoid self-loops
                ihat, jhat = _orient_edge(x, X[j], i, j, Y[group])
                ei.append(ihat)
                ej.append(jhat)

    return ei, ej


def plot_graph(G, X, groups, filename=None):
    """
    Plot the graph with node positions and cluster colors.

    Parameters:
        G (networkx.DiGraph): Directed graph.
        X (np.ndarray): Node positions.
        groups (np.ndarray): Cluster assignments.
        filename (str, optional): Filename to save the plot.
    """
    plt.figure(figsize=(6, 6))
    pos = {i: (X[i, 0], X[i, 1]) for i in range(len(X))}
    nx.draw(G, pos, node_size=50, node_color=groups, cmap=plt.cm.jet,
            edge_color="gray", alpha=0.7, with_labels=False)

    if filename:
        plt.savefig(filename, dpi=300)
    plt.show()


# Example usage with a fixed seed

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=1000)
parser.add_argument('--k', type=int, default=2) # number of clusters
parser.add_argument('--deg_scale', type=float, default=1)
parser.add_argument('--deg_shape', type=float, default=1)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

# Set a fixed seed for reproducibility
SEED = args.seed
np.random.seed(SEED)


#k_values = [2, 4, 8]  # Different cluster numbers

#for k in k_values:
G, X, groups = spatially_coherent_directed_graph(n=args.n, d=2, k=args.k, degreedist=lognorm(s=args.deg_shape, scale=np.exp(args.deg_scale)))
    #plot_graph(G, X, groups, filename=f"n-100-k-{k}.png")

    # Save edges to CSV
edge_list = np.array(G.edges()).T
    #np.savetxt(f"n-100-k-{k}.csv", edge_list, delimiter=",", fmt="%d")

save_path = '../data/spatial_graphs/scdg_n%d_k_%d_shape_%.3f_scale_%.3f_seed_%d.npz'%(args.n, args.k, args.deg_shape, args.deg_scale, args.seed)
np.savez(save_path, edge_index=edge_list, groups=groups)
