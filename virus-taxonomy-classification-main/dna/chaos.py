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
  
        # Aggiungi i nodi al grafo
        for label in np.unique(segments):
            self.graph_ohe.add_node(label)
            
            
        # Aggiungi gli archi tra i nodi adiacenti
        for y in range(segments.shape[0]):
            for x in range(segments.shape[1]):
                label1 = segments[y, x]
                for label2 in np.unique(segments):
                    if label1 != label2 and sono_adiacenti(segments, label1, label2, x, y):
                        self.graph_ohe.add_edge(label1, label2)

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

def visualize_graph(graph):
    pos = nx.spring_layout(graph)  # Define the layout (spring layout is just an example)

    # Draw nodes
    nx.draw_networkx_nodes(graph, pos, node_size=500, node_color='skyblue')

    # Draw edges
    nx.draw_networkx_edges(graph, pos, width=2, edge_color='gray')

    # Draw labels (optional)
    nx.draw_networkx_labels(graph, pos, font_size=10, font_color='black')

    # Display the graph
    plt.show()


def sono_adiacenti(segments, label1, label2, x, y):
    if x > 0 and segments[y, x - 1] == label2:
        return True
    if x < segments.shape[1] - 1 and segments[y, x + 1] == label2:
        return True
    if y > 0 and segments[y - 1, x] == label2:
        return True
    if y < segments.shape[0] - 1 and segments[y + 1, x] == label2:
        return True
    return False


if __name__ == '__main__':
    graph = ChaosGraph('ACTTCTTCACTTCTTCGGCGGC', 4)
