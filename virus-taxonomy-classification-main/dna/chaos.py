import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from skimage.segmentation import slic
from complexcgr import FCGR
from visualize import *

class ChaosGraph:
    def __init__(self, sequence: str, k_size: int):
        # Pulizia della sequenza
        valid_nucleotides = {'A', 'C', 'G', 'T', 'N'}
        self.sequence = ''.join([nuc if nuc in valid_nucleotides else 'N' for nuc in sequence])

        self.k_size = k_size

        #init grafo
        self.graph_chaos = nx.Graph()
        self.node_attr = []
        self.edge_attr = []
        
        # Generazione dell'immagine FCGR
        fcgr = FCGR(k=4, bits=8)
        chaos_rep = fcgr(self.sequence)
        chaos_image = fcgr.array2img(chaos_rep)
        image = np.array(chaos_image)
        
        # Segmentazione in superpixel
        num_superpixels = 10
        segments = slic(image, n_segments=num_superpixels, compactness=5, channel_axis=None)
        
        # Crea nodi
        attr = {}
        for y in range(segments.shape[0]):
            for x in range(segments.shape[1]):
                label = segments[y, x]
                if label not in self.graph_chaos:
                    node_feature = np.sum(segments == label)
                    attr[f'pixel_count'] = node_feature
                    self.graph_chaos.add_node(label, **attr)
        self.node_attr = list(attr.keys())

        # Crea archi 
        attr = {}
        pass
        self.edge_attr = list(attr.keys())

        #self.add_nodes_and_edges(segments)
        self.chaos_image = image
        self.segments = segments

    
if __name__ == '__main__':
    sequence1 = 'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'
    graph1 = ChaosGraph(sequence1, 8)
    print("---Grafo 1 ---")
    for node, attrs in graph1.graph_chaos.nodes(data=True):
        print(f"Nodo {node}: {attrs}")

    sequence2 = 'CCCCCCCCCCCCCCCCCCCCCCCCCTTTTAAAAAAAAAAAAAAAAAAAAAAAAA'
    graph2 = ChaosGraph(sequence2, 8)
    print("---Grafo 2 ---")
    for node, attrs in graph2.graph_chaos.nodes(data=True):
        print(f"Nodo {node}: {attrs}")

    sequence3 = 'ACTACTTCACTTCTTCACTTCTTCGGCGGCTTACTTCTCACTTCTCACTTCTTCGGCGGCCACTTCTTCGGCGGC'
    graph3 = ChaosGraph(sequence3, 8)
    print("---Grafo 3 ---")
    for node, attrs in graph3.graph_chaos.nodes(data=True):
        print(f"Nodo {node}: {attrs}")