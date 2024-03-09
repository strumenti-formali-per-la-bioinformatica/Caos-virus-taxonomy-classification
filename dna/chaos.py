import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from complexcgr import FCGR

class ChaosGraph:
    def __init__(self, sequence: str, k_size: int):
        # Pulizia della sequenza
        valid_nucleotides = {'A', 'C', 'G', 'T', 'N'}
        self.sequence = ''.join([nuc if nuc in valid_nucleotides else 'N' for nuc in sequence])

        self.k_size = k_size

        #init grafo
        self.graph_ohe = nx.Graph()
        self.graph = nx.Graph()
        self.node_attr = []
        self.edge_attr = []
        
        # Generazione dell'immagine FCGR
        fcgr = FCGR(k=k_size, bits=8)
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
                    attr["frequency"] = value
                    kmer = pixel2kmer_dict[(i+1,j+1)]
                    self.graph.add_node(kmer, value=kmer)
                    for i, k in enumerate(kmer):
                        for n in ['A', 'C', 'G', 'T']:
                            if k == n:
                                attr[f'{n}_{i}'] = 1
                            else:
                                attr[f'{n}_{i}'] = 0
                    self.graph_ohe.add_node(kmer, **attr)
        self.node_attr = list(attr.keys())
        
        attr = {}  
        # Crea archi
        for i, x in enumerate(self.graph_ohe.nodes(data=True)):
            for j, k in enumerate(list(self.graph_ohe.nodes(data=True))[i+1:]):
                kmer1, kmer2 = x[0], k[0]

                if kmer1 != kmer2:  # Evita di confrontare un k-mer con se stesso
                    distance = sum(ch1 != ch2 for ch1, ch2 in zip(kmer1, kmer2))
                    # Se la distanza soddisfa il criterio, ad esempio distanza di Hamming ≤ 1
                    if distance <= 2:
                        attr['distance'] = distance  
                        self.graph.add_edge(kmer1, kmer2, **attr)
                        self.graph_ohe.add_edge(kmer1, kmer2, **attr)
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


def get_frame(graph, framecount, pos):

    # Crea una figura per il plot
    plt.figure(figsize=(9, 9), dpi=150)

    # Disegna il grafo
    nx.draw(graph, pos, edge_color='black', width=1, linewidths=1,
            node_size=800, node_color='pink', alpha=0.9,
            labels={node: node for node in graph.nodes()})  # Usa la chiave del nodo come etichetta

    # Ottieni gli attributi 'distance' per ogni arco nel grafo
    edge_labels = nx.get_edge_attributes(graph, 'distance')

    # Disegna le etichette degli archi con la distanza
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color='red')

    # Nasconde gli assi per una visualizzazione più pulita
    plt.axis('off')
    plt.savefig(f"frame{framecount}.png")
    plt.show()

if __name__ == '__main__':
    sequence1 = 'ACTACTTCACTTCTTCACTTCTTCGGCGGCTTACTTCTCACTTCTCACTTCTTCGGCGGCCAC'
    graph1 = ChaosGraph(sequence1, 4)
    print("---Grafo 1 ---")
    for node, attrs in graph1.graph_ohe.nodes(data=True):
        print(f"Nodo {node}: {attrs}")
    graph1.plot_graph() 