def visualize_graph(self):
    pos = nx.spring_layout(self.graph_ohe)
    node_sizes = [self.graph_ohe.nodes[n].get('pixel_count', 1) * 100 for n in self.graph_ohe.nodes]  # Usa un valore di default se 'pixel_count' non è presente
    edge_colors = [self.graph_ohe[u][v].get('distance', 1) for u, v in self.graph_ohe.edges]  # Usa un valore di default per la distanza
    nx.draw(self.graph_ohe, pos, with_labels=True, node_size=node_sizes, edge_color=edge_colors, edge_cmap=plt.cm.Blues)
    plt.show()

def visualize_segmentation(self):
    """Visualizza l'immagine con le segmentazioni dei superpixel."""
    plt.imshow(mark_boundaries(self.chaos_image, self.segments))
    plt.title("Segmentazione Superpixel")
    plt.axis('off')
    plt.show()

def visualize_fcgr(self):
    """Visualizza la rappresentazione FCGR della sequenza."""
    plt.imshow(self.chaos_image, cmap='gray')
    plt.title("Rappresentazione FCGR")
    plt.axis('off')
    plt.show()


def visualize_intensity_graph(self):
    pos = {n: n for n in self.graph_intensity.nodes}
    node_colors = [self.graph_intensity.nodes[n]['intensity'] for n in self.graph_intensity.nodes]
    nx.draw(self.graph_intensity, pos, with_labels=False, node_size=50, node_color=node_colors, cmap=plt.cm.viridis)
    plt.title("Grafo basato su Intensità")
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