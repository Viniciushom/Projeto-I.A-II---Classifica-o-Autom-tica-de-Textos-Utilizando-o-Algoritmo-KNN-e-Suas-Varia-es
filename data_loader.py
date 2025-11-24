import numpy as np
import os

def load_dataset(dataset_name):
    # Função placeholder para carregar datasets. Para uso real, implemente leitura de arquivos CSV ou TXT.
    # Exemplo: Reuters, 20 Newsgroups, Ohsumed assumidos como listas de textos e labels.
    if dataset_name == 'toy':
        texts = ['this is a test document about sports', 'another doc on politics', 'sports news today', 'politics and economy']
        labels = [0, 1, 0, 1]  # 0: sports, 1: politics
        return texts, np.array(labels)
    # Para datasets reais, use: texts, labels = read_from_file(path)
    raise ValueError("Dataset não implementado. Baixe e adapte.")

