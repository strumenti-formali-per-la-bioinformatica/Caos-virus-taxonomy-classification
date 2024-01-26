import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from skimage.segmentation import slic, mark_boundaries
from complexcgr import FCGR

class ChaosGraph:
    def __init__(self, sequence: str, k_size: int):
        # Pulizia della sequenza
        valid_nucleotides = {'A', 'C', 'G', 'T', 'N'}
        self.sequence = ''.join([nuc if nuc in valid_nucleotides else 'N' for nuc in sequence])
        self.k_size = k_size
        self.graph_ohe = nx.Graph()
        
        # Generazione dell'immagine FCGR
        fcgr = FCGR(k=4, bits=8)
        chaos_rep = fcgr(self.sequence)
        chaos_image = fcgr.array2img(chaos_rep)
        image = np.array(chaos_image)
        
        # Segmentazione in superpixel
        num_superpixels = 10
        segments = slic(image, n_segments=num_superpixels, compactness=5, channel_axis=None)
        
        # Aggiunta nodi e archi
        self.add_nodes_and_edges(segments)
        self.chaos_image = image
        self.segments = segments

    def visualize_fcgr(self):
        """Visualizza la rappresentazione FCGR della sequenza."""
        plt.imshow(self.chaos_image, cmap='gray')
        plt.title("Rappresentazione FCGR")
        plt.axis('off')
        plt.show()

    def visualize_segmentation(self):
        """Visualizza l'immagine con le segmentazioni dei superpixel."""
        plt.imshow(mark_boundaries(self.chaos_image, self.segments))
        plt.title("Segmentazione Superpixel")
        plt.axis('off')
        plt.show()

    def add_nodes_and_edges(self, segments):
        for y in range(segments.shape[0]):
            for x in range(segments.shape[1]):
                label = segments[y, x]
                if label not in self.graph_ohe:
                    node_feature = np.sum(segments == label)
                    self.graph_ohe.add_node(label, pixel_count=node_feature)
                self.add_edges(segments, label, x, y)

    def add_edges(self, segments, label1, x, y):
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < segments.shape[1] and 0 <= ny < segments.shape[0]:
                label2 = segments[ny, nx]
                if label1 != label2 and not self.graph_ohe.has_edge(label1, label2):
                    edge_feature = np.sqrt(dx**2 + dy**2)
                    self.graph_ohe.add_edge(label1, label2, distance=edge_feature)

    def visualize_graph(self):
        pos = nx.spring_layout(self.graph_ohe)
        node_sizes = [self.graph_ohe.nodes[n].get('pixel_count', 1) * 100 for n in self.graph_ohe.nodes]  # Usa un valore di default se 'pixel_count' non è presente
        edge_colors = [self.graph_ohe[u][v].get('distance', 1) for u, v in self.graph_ohe.edges]  # Usa un valore di default per la distanza
        nx.draw(self.graph_ohe, pos, with_labels=True, node_size=node_sizes, edge_color=edge_colors, edge_cmap=plt.cm.Blues)
        plt.show()
   
    def add_intensity_based_graph(self, threshold=0.5):
        self.graph_intensity = nx.Graph()
        
        # Normalizzazione dell'immagine per facilitare il confronto con il threshold
        normalized_image = self.chaos_image / np.max(self.chaos_image)
        
        # Crea nodi per celle con intensità superiore al threshold
        for y in range(normalized_image.shape[0]):
            for x in range(normalized_image.shape[1]):
                if normalized_image[y, x] > threshold:
                    self.graph_intensity.add_node((x, y), intensity=normalized_image[y, x])

        # Aggiungi archi basati sulla vicinanza spaziale
        for y in range(normalized_image.shape[0]):
            for x in range(normalized_image.shape[1]):
                if (x, y) in self.graph_intensity.nodes:
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        next_x, next_y = x + dx, y + dy  # Rinomina nx e ny in next_x e next_y
                        if (next_x, next_y) in self.graph_intensity.nodes:
                            self.graph_intensity.add_edge((x, y), (next_x, next_y))

    def visualize_intensity_graph(self):
        pos = {n: n for n in self.graph_intensity.nodes}
        node_colors = [self.graph_intensity.nodes[n]['intensity'] for n in self.graph_intensity.nodes]
        nx.draw(self.graph_intensity, pos, with_labels=False, node_size=50, node_color=node_colors, cmap=plt.cm.viridis)
        plt.title("Grafo basato su Intensità")
        plt.show()

if __name__ == '__main__':
    sequence1 = 'ACTTCTTCACTTCTTCGGCGGC'
    graph1 = ChaosGraph(sequence1, 4)
    graph1.visualize_fcgr()
    graph1.visualize_segmentation()
    graph1.visualize_graph()
    graph1.add_intensity_based_graph(threshold=0.5)
    graph1.visualize_intensity_graph()

    sequence2 = 'ACTTCACTTCTTCACTTCTTCGGCGGCTTACTTCTTCACTTCTTCGGCGGCCACTTCTTCGGCGGC'
    graph2 = ChaosGraph(sequence2, 4)
    graph2.visualize_fcgr()
    graph2.visualize_segmentation()
    graph2.visualize_graph() 
    graph2.add_intensity_based_graph(threshold=0.5)
    graph2.visualize_intensity_graph()

    sequence3 = 'ACTACTTCACTTCTTCACTTCTTCGGCGGCTTACTTCTTCACTTCTTCGGCGGCCACTTCTTCGGCGGCTCACTTCTTCACTTCTTCGGCGGCTTACTTCTTCACTTCTTCGGCGGCCACTTCTTCGGCGGC'
    graph3 = ChaosGraph(sequence3, 4)
    graph3.visualize_fcgr()
    graph3.visualize_segmentation()
    graph3.visualize_graph()
    graph3.add_intensity_based_graph(threshold=0.5)
    graph3.visualize_intensity_graph()