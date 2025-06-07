import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
def shortest_path_distance(edge_index):
    G = nx.DiGraph()
    G.add_edges_from([(int(edge[0]), int(edge[1])) for edge in edge_index.T])
    spd = dict(nx.all_pairs_shortest_path_length(G))
    return spd


def num_walks(edge_index, length=10):
    G = nx.DiGraph()
    G.add_edges_from([(int(edge[0]), int(edge[1])) for edge in edge_index.T])
    num = nx.number_of_walks(G, length)
    return num

def draw_graph(edge_index, seed=0):
    G = nx.DiGraph()
    G.add_edges_from([(edge[0].item(), edge[1].item()) for edge in edge_index.T])
    pos = nx.spring_layout(G, seed=seed)

    node_sizes = [3 + 10 * i for i in range(len(G))]
    M = G.number_of_edges()
    edge_colors = range(2, M + 2)
    edge_alphas = [(5 + i) / (M + 4) for i in range(M)]
    cmap = plt.cm.plasma

    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="indigo")
    edges = nx.draw_networkx_edges(
        G,
        pos,
        node_size=node_sizes,
        arrowstyle="->",
        arrowsize=10,
        edge_color=edge_colors,
        edge_cmap=cmap,
        width=2,
    )

    # set alpha value for each edge
    for i in range(M):
        edges[i].set_alpha(edge_alphas[i])

    pc = mpl.collections.PatchCollection(edges, cmap=cmap)
    pc.set_array(edge_colors)

    ax = plt.gca()
    ax.set_axis_off()
    plt.colorbar(pc, ax=ax)
    plt.show()

    return G


