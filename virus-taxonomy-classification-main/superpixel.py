from skimage.segmentation import slic
from skimage.color import label2rgb
import numpy as np
import cv2
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def crea_superpixel(image_path, output_path, n_segments, compactness):
    # Carica l'immagine usando OpenCV
    image = cv2.imread(image_path)

    # Applica la segmentazione SLIC per creare superpixel
    segments = slic(image, n_segments=n_segments, compactness=compactness)

    # Visualizza i superpixel con l'overlay sull'immagine originale
    superpixel_image = label2rgb(segments, image=image, kind='avg')

    # Converti l'immagine risultante in bianco e nero
    # Nota: la moltiplicazione per 255 e la conversione in uint8 è necessaria
    # per convertire i valori da un intervallo [0, 1] a [0, 255]
    superpixel_image_gray = cv2.cvtColor((superpixel_image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    # Salva l'immagine risultante in bianco e nero
    cv2.imwrite(output_path, superpixel_image_gray)
    return segments

segments =crea_superpixel('/home/alessiature/Documenti/Caos-virus-taxonomy/virus-taxonomy-classification-main/img/file_50_AC_000192.1.jpg',
                'superpixel_image_modified_bw.jpg', n_segments=200, compactness=5)

#Piccolo esempio per vedere se va un grafo
# Funzione per controllare se due superpixel sono adiacenti
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

# Crea un grafo
G = nx.Graph()

# Aggiungi i nodi al grafo
for label in np.unique(segments):
    G.add_node(label)

# Aggiungi gli archi tra i nodi adiacenti
for y in range(segments.shape[0]):
    for x in range(segments.shape[1]):
        label1 = segments[y, x]
        for label2 in np.unique(segments):
            if label1 != label2 and sono_adiacenti(segments, label1, label2, x, y):
                G.add_edge(label1, label2)

print("Nodi nel grafo:", G.number_of_nodes())
print("Archi nel grafo:", G.number_of_edges())
