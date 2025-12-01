import os
import time
from collections import Counter

import numpy as np
from data_loader import _parse_reuters_sgml_file

# Processa todos os arquivos .sgm na pasta atual, um a um, imprimindo progresso
cwd = os.getcwd()
files = [f for f in sorted(os.listdir(cwd)) if f.lower().endswith('.sgm') or f.lower().startswith('reut')]
if not files:
    print('Nenhum arquivo .sgm encontrado no diretório atual')
    raise SystemExit(1)

label_map = {}
all_texts = []
all_labels = []
start_total = time.time()

for i, fname in enumerate(files, 1):
    fpath = os.path.join(cwd, fname)
    t0 = time.time()
    before = len(all_texts)
    try:
        _parse_reuters_sgml_file(fpath, label_map, all_texts, all_labels)
    except Exception as e:
        print(f'Erro ao parsear {fname}:', e)
        continue
    after = len(all_texts)
    elapsed = time.time() - t0
    print(f'[{i}/{len(files)}] {fname}: +{after-before} docs (total {after}) — {elapsed:.2f}s')

print('\nResumo final:')
print('Arquivos processados:', len(files))
print('Documentos totais extraídos:', len(all_texts))
print('Categorias encontradas:', len(label_map))
print('Top 10 categorias por frequência:')
print(Counter(all_labels).most_common(10))
print('Tempo total: %.2fs' % (time.time() - start_total))

# Opcional: rodar pipeline curta sobre os primeiros 200 documentos para verificar etapas posteriores
N = min(200, len(all_texts))
if N == 0:
    print('Sem documentos para pipeline curta.')
    raise SystemExit(0)

from preprocess import preprocess_texts, build_vocabulary
from tfidf import compute_tfidf
from feature_generation import generate_similarity_features
from knn_variants import standard_knn

print('\nExecutando pipeline curta sobre primeiros %d documentos...' % N)
texts = all_texts[:N]
labels = np.array(all_labels[:N])

t0 = time.time()
processed = preprocess_texts(texts)
print('Preprocessado em %.2fs' % (time.time()-t0))

t0 = time.time()
vocab = build_vocabulary(processed)
print('Vocabulário de tamanho', len(vocab), 'em %.2fs' % (time.time()-t0))

t0 = time.time()
vectors = compute_tfidf(processed, vocab)
print('TF-IDF shape:', vectors.shape, 'em %.2fs' % (time.time()-t0))

if vectors.shape[0] > 1:
    t0 = time.time()
    X_train = vectors[:-1]
    new_features = generate_similarity_features(X_train)
    print('Gerou features shape', new_features.shape, 'em %.2fs' % (time.time()-t0))

    pred = standard_knn(X_train, labels[:-1], vectors[-1:].reshape(1, -1), k=3)
    print('Predição KNN exemplo (último doc):', pred)

print('Pipeline curta concluída.')
