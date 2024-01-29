import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from math import sqrt
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
                        if n == get_this_sequence_letter(j , i):
                            attr[f'{n}'] = 1
                        else:
                            attr[f'{n}'] = 0
                # e la posizione in termini spaziali
                    attr["position"] = (i,j)
                    attr["value"] = value
                    #trovare un nome migliore univoco
                    self.graph_chaos.add_node((i,j), **attr)
        self.node_attr = list(attr.keys())

        # Crea archi
        # archi tra tutti i nodi
        for i, x in enumerate(self.graph_chaos.nodes(data=True)):
            for j , k  in enumerate(list(self.graph_chaos.nodes(data=True))[i+1:]):
                node1, node2 = x[0], k[0]
                first_node_position = x[1]['position']
                second_node_position = k[1]['position']
                self.graph_chaos.add_edge(node1, node2)
        # feature per archi la distanza euclidea  tra i nodi che li compongono
                pass

    def plot_graph(self):
       pos = nx.circular_layout(self.graph_chaos)
       plt.figure()
       nx.draw(
            self.graph_chaos,
            pos,
            edge_color='black',
            width=1,
            linewidths=1,
            node_size=1000,
            node_color='pink',
            alpha=0.9,
            labels={node[0]: node[1] for node in list(self.graph_chaos.nodes.data("value"))}
        )
       value_labels = nx.get_edge_attributes(self.graph_chaos, 'value')
       frequency_labels = nx.get_edge_attributes(self.graph_chaos, 'distance')
       edge_labels = {key: f'{value_labels[key]} - {frequency_labels[key]}' for key in value_labels}
       nx.draw_networkx_edge_labels(
            self.graph_chaos,
            pos,
            edge_labels=edge_labels
        )
       plt.axis('off')
       plt.show()

def get_this_sequence_letter(x_axis_position, y_axis_position):
    if y_axis_position < 8 :
        if x_axis_position < 8:
            return 'C'
        else :
            return 'A'
    else:
        if x_axis_position < 8:
            return 'G'
        else :
            return 'T'

if __name__ == '__main__':
    sequence1 = 'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'
    graph1 = ChaosGraph(sequence1, 8)
    print("---Grafo 1 ---")
    for node, attrs in graph1.graph_chaos.nodes(data=True):
        print(f"Nodo {node}: {attrs}")
    graph1.plot_graph()

    sequence2 = 'CCCCCCCCCCCCCCCCCCCCCCCCCTTTTAAAAAAAAAAAAAAAAAAAAAAAAA'
    graph2 = ChaosGraph(sequence2, 8)
    print("---Grafo 2 ---")
    for node, attrs in graph2.graph_chaos.nodes(data=True):
        print(f"Nodo {node}: {attrs}")
    graph2.plot_graph()

    sequence3 = 'ACTACTTCACTTCTTCACTTCTTCGGCGGCTTACTTCTCACTTCTCACTTCTTCGGCGGCCACTTCTTCGGCGGC'
    graph3 = ChaosGraph(sequence3, 8)
    print("---Grafo 3 ---")
    for node, attrs in graph3.graph_chaos.nodes(data=True):
        print(f"Nodo {node}: {attrs}")
    graph3.plot_graph()    