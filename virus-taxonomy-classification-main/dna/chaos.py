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
        chaos_array = np.asarray(chaos_rep)
        # C_quadrant = chaos_array[:8,:8]
        # G_quadrant = chaos_array[8:16,:8]
        # A_quadrant = chaos_array[:8,8:16]
        # T_quadrant = chaos_array[8:16,8:16]

        # print(chaos_array)
        # print("C QUADRANT")
        # print(C_quadrant)
        # print("A QUADRANT")
        # print(A_quadrant)
        # print("G QUADRANT")
        # print(G_quadrant)
        # print("T QUADRANT")
        # print(T_quadrant)

        chaos_image = fcgr.array2img(chaos_rep)
        image = np.array(chaos_image)
        
        # Segmentazione in superpixel
        num_superpixels = 10
        segments = slic(image, n_segments=num_superpixels, compactness=5, channel_axis=None)
        
        # Crea nodi
        x = chaos_array.shape[0]
        y = chaos_array.shape[1]

        attr = {}

        # for ogni x e y se [x,y] non è uguale a 0 allora lo consideriamo un nodo con valore [x,y]
        for i in range(x):
            for j in range(y):
                value = chaos_array[i,j]
                if value != 0:
                # come attributo la lettera che rappresenta in one hot encoding
                    for n in ['A', 'C', 'G', 'T']:
                        if n == get_this_sequence_letter():
                            attr[f'{n}'] = 1
                        else:
                            attr[f'{n}'] = 0
                # e la posizione in termini spaziali
                    attr["position"] = (i,j)
                    self.graph_chaos.add_node(value, **attr)

        self.node_attr = list(attr.keys())

        # Crea archi
        # archi tra tutti i nodi
        for i, x in enumerate(self.graph_chaos.nodes):
            for j , k in enumerate(list(self.graph_chaos.nodes)[i+1:]):
                self.graph_chaos.add_edge(x, k)
                pass
        # feature per archi la distanza euclidea  tra i nodi che li compongono

        #self.add_nodes_and_edges(segments)
        self.chaos_image = image
        self.segments = segments

    
def get_this_sequence_letter():
    return 'A'

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