import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from math import sqrt
from skimage.segmentation import slic
from complexcgr import FCGR

class ChaosGraph:
    def __init__(self, sequence: str, k_size: int):
        # Pulizia della sequenza
        valid_nucleotides = {'A', 'C', 'G', 'T', 'N'}
        self.sequence = ''.join([nuc if nuc in valid_nucleotides else 'N' for nuc in sequence])

        self.k_size = k_size

        #init grafo
        self.graph_chaos = nx.Graph()
        self.graph = nx.Graph()
        self.node_attr = []
        self.edge_attr = []
        
        # Generazione dell'immagine FCGR
        fcgr = FCGR(k=4, bits=8)
        chaos_rep = fcgr(self.sequence)
        chaos_array = np.asarray(chaos_rep)
      
        # Crea nodi
        x = chaos_array.shape[0]
        y = chaos_array.shape[1]

        # Reverse the dictionary to create a pixel2kmer dictionary
        pixel2kmer_dict = {v: k for k, v in fcgr.kmer2pixel.items()}
        attr = {}
        # for ogni x e y se [x,y] non è uguale a 0 allora lo consideriamo un nodo con valore [x,y]
        for i in range(x):
            for j in range(y):
                value = chaos_array[i,j]
                if value != 0:
                    attr["position"] = (i,j)
                    attr["value"] = value
                    kmer = pixel2kmer_dict[(i+1,j+1)]
                    self.graph.add_node(kmer, value=kmer)
                    for i, k in enumerate(kmer):
                        for n in ['A', 'C', 'G', 'T']:
                            if k == n:
                                attr[f'{n}_{i}'] = 1
                            else:
                                attr[f'{n}_{i}'] = 0
                    #trovare un nome migliore univoco
                    self.graph_chaos.add_node(kmer, **attr)
        self.node_attr = list(attr.keys())
        
        # Crea archi
        # archi tra tutti i nodi
        attr={}
        for i, x in enumerate(self.graph_chaos.nodes(data=True)):
            for j , k  in enumerate(list(self.graph_chaos.nodes(data=True))[i+1:]):
                node1, node2 = x[0], k[0]
                first_node_position = x[1]['position']
                second_node_position = k[1]['position']
                distance = sqrt((second_node_position[0] - first_node_position[0]) ** 2 + (second_node_position[1] - first_node_position[1]) ** 2)
                attr['distance']= distance
                kmer1 = pixel2kmer_dict[(first_node_position[0] +1,first_node_position[1]+1)]
                kmer2 = pixel2kmer_dict[(second_node_position[0] +1,second_node_position[1]+1)]
                self.graph.add_edge(kmer1,kmer2,**attr)
                self.graph_chaos.add_edge(node1, node2, **attr)
        self.edge_attr = list(attr.keys())

    def plot_graph(self):
        # Definisci la posizione dei nodi nel grafo
        pos = nx.circular_layout(self.graph)
        # Crea una figura per il plot
        plt.figure(figsize=(8, 8))

        # Disegna il grafo
        nx.draw(self.graph, pos, edge_color='black', width=1, linewidths=1,
                node_size=500, node_color='pink', alpha=0.9,
                labels={node: node for node in self.graph.nodes()})  # Usa la chiave del nodo come etichetta

        # Ottieni gli attributi 'distance' per ogni arco nel grafo
        edge_labels = nx.get_edge_attributes(self.graph, 'distance')

        # Disegna le etichette degli archi con la distanza
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_color='red')

        # Nasconde gli assi per una visualizzazione più pulita
        plt.axis('off')
        plt.show()

if __name__ == '__main__':
    sequence1 = 'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'
    graph1 = ChaosGraph(sequence1, 4)
    print("---Grafo 1 ---")
    for node, attrs in graph1.graph_chaos.nodes(data=True):
        print(f"Nodo {node}: {attrs}")
    graph1.plot_graph()

    sequence2 = 'CCCCCCCCCCCCCCCCCCCCCCCCCTTTTAAAAAAAAAAAAAAAAAAAAAAAAA'
    graph2 = ChaosGraph(sequence2,4)
    print("---Grafo 2 ---")
    for node, attrs in graph2.graph_chaos.nodes(data=True):
        print(f"Nodo {node}: {attrs}")
    graph2.plot_graph()    

    sequence3 = 'ACTACTTCACTTCTTCACTTCTTCGGCGGCTTACTTCTCACTTCTCACTTCTTCGGCGGCCACTTCTTCGGCGGC'
    graph3 = ChaosGraph(sequence3, 4)
    print("---Grafo 3 ---")
    for node, attrs in graph3.graph_chaos.nodes(data=True):
        print(f"Nodo {node}: {attrs}") 
    graph3.plot_graph()    