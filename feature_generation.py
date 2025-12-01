import numpy as np
from scipy.spatial.distance import cdist

def generate_similarity_features(X_train, k=5):
    # Gera matriz de similaridade e novas features (ex.: grau médio por classe no grafo kNN)
    sim_matrix = 1 - cdist(X_train, X_train, 'cosine')
    np.fill_diagonal(sim_matrix, 0)  # Sem auto-similaridade
    # Exemplo de feature: número de vizinhos similares (grau no grafo kNN)
    degrees = np.sum(sim_matrix > np.sort(sim_matrix, axis=1)[:, -k-1], axis=1)  # Top k por linha
    return degrees.reshape(-1, 1)  # Adicione mais features conforme necessário