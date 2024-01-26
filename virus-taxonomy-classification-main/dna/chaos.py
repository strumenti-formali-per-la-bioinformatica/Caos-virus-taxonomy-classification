import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from skimage.segmentation import slic, mark_boundaries
from complexcgr import FCGR
from visualize import *

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
        
        # Crea nodi
        for y in range(segments.shape[0]):
            for x in range(segments.shape[1]):
                label = segments[y, x]
                if label not in self.graph_ohe:
                    node_feature = np.sum(segments == label)
                    self.graph_ohe.add_node(label, pixel_count=node_feature)
        # Crea archi 
        pass
        self.add_nodes_and_edges(segments)
        self.chaos_image = image
        self.segments = segments

    def add_edges(self, segments, label1, x, y):
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < segments.shape[1] and 0 <= ny < segments.shape[0]:
                label2 = segments[ny, nx]
                if label1 != label2 and not self.graph_ohe.has_edge(label1, label2):
                    edge_feature = np.sqrt(dx**2 + dy**2)
                    self.graph_ohe.add_edge(label1, label2, distance=edge_feature)


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