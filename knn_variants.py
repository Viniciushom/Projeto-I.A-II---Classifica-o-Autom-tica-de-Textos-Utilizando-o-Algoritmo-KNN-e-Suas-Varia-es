import numpy as np
from scipy.spatial.distance import cdist
from collections import Counter

def cosine_similarity(X, Y):
    return 1 - cdist(X, Y, 'cosine')

def standard_knn(X_train, y_train, X_test, k):
    sim = cosine_similarity(X_test, X_train)[0]  # Similaridades (maior é melhor)
    indices = np.argsort(-sim)[:k]  # Top k mais similares
    labels = y_train[indices]
    return Counter(labels).most_common(1)[0][0]

def kinn(X_train, y_train, X_test, k):
    n_train = len(y_train)
    sim_test_to_train = cosine_similarity(X_test, X_train)[0]  # Sim de test a train
    inverse_neighbors = []
    for i in range(n_train):
        # Similaridades de train_i a outros trains (excluindo si mesmo)
        sim_i_to_others = cosine_similarity(X_train[i:i+1], np.delete(X_train, i, axis=0))[0]
        kth_sim = np.sort(sim_i_to_others)[-k] if k <= n_train - 1 else np.min(sim_i_to_others)
        sim_i_to_test = sim_test_to_train[i]
        if sim_i_to_test >= kth_sim:  # Se test está entre os k mais similares para train_i
            inverse_neighbors.append(i)
    if not inverse_neighbors:
        return standard_knn(X_train, y_train, X_test, k)  # Fallback
    labels = y_train[inverse_neighbors]
    return Counter(labels).most_common(1)[0][0]

def ksnn(X_train, y_train, X_test, k):
    # Combinação: interseção de KNN e kINN
    sim = cosine_similarity(X_test, X_train)[0]
    knn_indices = set(np.argsort(-sim)[:k])
    inverse_indices = set()
    n_train = len(y_train)
    for i in range(n_train):
        sim_i_to_others = cosine_similarity(X_train[i:i+1], np.delete(X_train, i, axis=0))[0]
        kth_sim = np.sort(sim_i_to_others)[-k] if k <= n_train - 1 else np.min(sim_i_to_others)
        sim_i_to_test = sim[i]
        if sim_i_to_test >= kth_sim:
            inverse_indices.add(i)
    symmetric_indices = knn_indices.intersection(inverse_indices)
    if not symmetric_indices:
        return standard_knn(X_train, y_train, X_test, k)
    labels = y_train[list(symmetric_indices)]
    return Counter(labels).most_common(1)[0][0]