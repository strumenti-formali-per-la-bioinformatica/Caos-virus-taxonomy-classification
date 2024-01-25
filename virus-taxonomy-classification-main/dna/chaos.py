import matplotlib.pyplot as plt
from complexcgr import FCGR
import networkx as nx
import numpy as np
from skimage.segmentation import slic, mark_boundaries

class ChaosGraph:
    def __init__(self, sequence: str, k_size: int):
        # save sequence and k-mer size
        self.sequence = sequence
        self.k_size = k_size

        # get kmers
        n_kmers: int = len(sequence) - k_size + 2
        self.kmers = [self.sequence[i:i + k_size - 1] for i in range(n_kmers)]

        # init graph
        self.graph = nx.DiGraph()
        self.graph_ohe = nx.DiGraph()
        self.node_attr = []
        self.edge_attr = []

        # Convert sequence to fcgr representation with corresponding k size
        fcgr = FCGR(k=8, bits=16)
        chaos_rep = fcgr(sequence)
        chaos_image = fcgr.array2img(chaos_rep)
        # This gives us a PIL image
        chaos_image.show()
        # Convert Pillow image to NumPy array
        image = np.array(chaos_image)
        # Define the number of desired superpixels
        num_superpixels = 200

        # Perform superpixel segmentation
        segments = slic(image, n_segments=num_superpixels, compactness=5, channel_axis = None)

        # Visualize the original image with superpixel boundaries
        superpixel_boundaries = mark_boundaries(image, segments)
        plt.imshow(superpixel_boundaries)
        plt.axis('off')
        plt.show()

        # create nodes
        attr = {}
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
        for i in range(len(self.kmers) - 1):
            k1, k2 = self.kmers[i], self.kmers[i + 1]
            # if edge already exist, update frequency
            if self.graph.has_edge(k1, k2):
                self.graph[k1][k2]['frequency'] += 1
                self.graph_ohe[k1][k2]['frequency'] += 1
            # else add it
            else:
                self.graph.add_edge(k1, k2, value=k2[-1], frequency=1)
                # create an attribute for each possible nucleotide
                for n in ['A', 'C', 'G', 'T']:
                    if k2[-1] == n:
                        attr[n] = 1
                    else:
                        attr[n] = 0
                self.graph_ohe.add_edge(k1, k2, frequency=1, **attr)
        self.edge_attr = list(attr.keys())
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
            node_size=1000,
            node_color='pink',
            alpha=0.9,
            labels={node[0]: node[1] for node in list(self.graph.nodes.data("value"))}
        )
        value_labels = nx.get_edge_attributes(self.graph, 'value')
        frequency_labels = nx.get_edge_attributes(self.graph, 'frequency')
        edge_labels = {key: f'{value_labels[key]} - {frequency_labels[key]}' for key in value_labels}
        nx.draw_networkx_edge_labels(
            self.graph,
            pos,
            edge_labels=edge_labels
        )
        plt.axis('off')
        plt.show()


if __name__ == '__main__':
    graph = ChaosGraph('ACTTCTTCACTTCTTCGGCGGC', 4)
    graph.plot_graph()
