"""
load_vectors.py: Carregar vetores TF-IDF salvos e executar análise/classificação
Requer ter rodado save_vectors.py uma vez.
"""
import os
import numpy as np
from scipy.sparse import load_npz
from collections import Counter
from knn_variants import standard_knn, kinn, ksnn

input_file = 'reuters_vectors.npz'
metadata_file = 'reuters_metadata.npz'

if not os.path.exists(input_file) or not os.path.exists(metadata_file):
    raise FileNotFoundError(f'Arquivos não encontrados. Execute save_vectors.py primeiro.')

print(f'Carregando vetores de {input_file}...')
vectors_sparse = load_npz(input_file)
vectors = vectors_sparse.toarray()  # Converter para dense se necessário para análise

print(f'Carregando metadados de {metadata_file}...')
meta = np.load(metadata_file, allow_pickle=True)
labels = meta['labels']
vocab = meta['vocab']
doc_count = int(meta['doc_count'])
vocab_size = int(meta['vocab_size'])

print(f'\nResumo do dataset:')
print(f'  Documentos: {doc_count}')
print(f'  Vocabulário: {vocab_size}')
print(f'  Vetores shape: {vectors.shape}')
print(f'  Categorias únicas: {len(np.unique(labels))}')
print(f'  Top 5 categorias: {Counter(labels).most_common(5)}')

# Exemplo: rodar classificações KNN sobre uma amostra de teste
if vectors.shape[0] < 2:
    print('Erro: não há dados suficientes')
    raise SystemExit(1)

# Dividir train/test simples (últimos 10 documentos como teste)
n_test = min(10, vectors.shape[0] // 10)
X_train = vectors[:-n_test]
X_test = vectors[-n_test:]
y_train = labels[:-n_test]
y_test = labels[-n_test:]

print(f'\nTestando KNN em {n_test} documentos...')
k = 5
correct_knn = 0
correct_kinn = 0
correct_ksnn = 0

for i in range(X_test.shape[0]):
    test_vec = X_test[i:i+1]
    true_label = y_test[i]
    
    try:
        pred_knn = standard_knn(X_train, y_train, test_vec, k)
        if pred_knn == true_label:
            correct_knn += 1
    except Exception as e:
        print(f'  Erro KNN no doc {i}: {e}')
    
    try:
        pred_kinn = kinn(X_train, y_train, test_vec, k)
        if pred_kinn == true_label:
            correct_kinn += 1
    except Exception as e:
        print(f'  Erro kINN no doc {i}: {e}')
    
    try:
        pred_ksnn = ksnn(X_train, y_train, test_vec, k)
        if pred_ksnn == true_label:
            correct_ksnn += 1
    except Exception as e:
        print(f'  Erro kSNN no doc {i}: {e}')

print(f'\nResultados (accuracy em {n_test} docs):')
print(f'  KNN:  {correct_knn}/{n_test} = {100*correct_knn/n_test:.1f}%')
print(f'  kINN: {correct_kinn}/{n_test} = {100*correct_kinn/n_test:.1f}%')
print(f'  kSNN: {correct_ksnn}/{n_test} = {100*correct_ksnn/n_test:.1f}%')

print('\n✓ Análise concluída.')
