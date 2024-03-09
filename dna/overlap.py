import matplotlib.pyplot as plt
import networkx as nx


class OverlapGraph:
    def __init__(self, sequence: str, k_size: int):
        # save sequence and k-mer size
        self.sequence = sequence
        self.k_size = k_size

        # get kmers
        n_kmers: int = len(sequence) - k_size + 1
        self.kmers = [self.sequence[i:i + k_size] for i in range(n_kmers)]

        # init graph
        self.graph = nx.DiGraph()
        self.graph_ohe = nx.DiGraph()
        self.node_attr = []
        self.edge_attr = []

        attr = {}

        # create nodes
        for kmer in self.kmers:
            self.graph.add_node(kmer, value=kmer)
            # create an attribute for each possible nucleotide for each nucleotide in kmer
            for i, k in enumerate(kmer):
                for n in ['A', 'C', 'G', 'T']:
                    if k == n:
                        attr[f'{n}_{i}'] = 1
                    else:
                        attr[f'{n}_{i}'] = 0
            self.graph_ohe.add_node(kmer, **attr)

        self.node_attr = list(attr.keys())

        attr = {}
        # create edges
        for k1 in self.kmers:
            for kmer in self.kmers:
                if k1 != kmer:
                    overlap_size = get_overlap_size(kmer, k1)
                    if overlap_size >= 4:
                        # If the arc isn't present add it, if it is keep track of how many times we would've added it
                        # if edge already exist, update frequency
                        if self.graph.has_edge(k1, kmer):
                            self.graph[k1][kmer]['frequency'] += 1
                            self.graph_ohe[k1][kmer]['frequency'] += 1
                        # else add it
                        else:
                            self.graph.add_edge(k1, kmer, value=overlap_size, frequency = 1)
                            self.graph_ohe.add_edge(k1, kmer, value=overlap_size, frequency = 1)

        # Use the frequency of a type of arc as a feature
        self.edge_attr.append('frequency')
            
    def plot_graph(self):
        pos = nx.circular_layout(self.graph)
        plt.figure()
        nx.draw(
            self.graph,
            pos,
            edge_color='black',
            width=1,
            linewidths=1,
            node_size=2000,
            node_color='pink',
            alpha=0.9,
            connectionstyle='arc3, rad = 0.1',
            labels={node[0]: node[1] for node in list(self.graph.nodes.data("value"))}
        )
        value_labels = nx.get_edge_attributes(self.graph, 'value')
        edge_labels = {key: f'{value_labels[key]}' for key in value_labels}
        nx.draw_networkx_edge_labels(
            self.graph,
            pos,
            edge_labels=edge_labels,
            verticalalignment = "center",
            label_pos=0.43,
            bbox=dict(alpha=0)
        )
        plt.axis('off')
        plt.show()


def compute_lps(pattern):
    m = len(pattern)
    lps = [0] * m
    j = 0

    for i in range(1, m):
        while j > 0 and pattern[i] != pattern[j]:
            j = lps[j - 1]

        if pattern[i] == pattern[j]:
            j += 1

        lps[i] = j

    return lps

def get_overlap_size(string1, string2):
    combined = string1 + '$' + string2
    lps = compute_lps(combined)

    overlap_length = lps[-1]

    return overlap_length


def get_frame(graph, framecount, pos):
    plt.figure(figsize=(12, 9), dpi=150)
    nx.draw(
        graph,
        pos,
        edge_color='black',
        width=1,
        linewidths=1,
        node_size=2000,
        node_color='pink',
        alpha=0.9,
        connectionstyle='arc3, rad = 0.1',
        labels={node[0]: node[1] for node in list(graph.nodes.data("value"))}
    )
    value_labels = nx.get_edge_attributes(graph, 'value')
    edge_labels = {key: f'{value_labels[key]}' for key in value_labels}
    nx.draw_networkx_edge_labels(
        graph,
        pos,
        edge_labels=edge_labels,
        verticalalignment = "center",
        label_pos=0.43,
        bbox=dict(alpha=0)
    )
    plt.axis('off')
    plt.savefig(f"frame{framecount}.png")
    plt.show()
    plt.close()
    


if __name__ == '__main__':
    graph = OverlapGraph('GTACGTACGAT', 6)
    graph.plot_graph()
