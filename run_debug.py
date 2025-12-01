import os
import time
from data_loader import load_dataset
from preprocess import preprocess_texts, build_vocabulary
from tfidf import compute_tfidf
from knn_variants import standard_knn
from feature_generation import generate_similarity_features

# Debug: processa apenas os primeiros N documentos
N = 100

cwd = os.getcwd()
# tenta carregar apenas o primeiro arquivo .sgm disponível
files = [f for f in os.listdir(cwd) if f.lower().endswith('.sgm') or f.lower().startswith('reut')]
if not files:
    print('Nenhum arquivo .sgm encontrado no diretório atual')
    raise SystemExit(1)

fpath = os.path.join(cwd, files[0])
print('Usando arquivo:', files[0])
texts, labels = load_dataset('reuters', fpath)
print('Total documentos neste arquivo:', len(texts))
texts = texts[:N]
labels = labels[:N]
print(f'Processando primeiros {len(texts)} documentos')

start = time.time()
processed = preprocess_texts(texts)
print('Preprocessado em %.2fs' % (time.time()-start))

start = time.time()
vocab = build_vocabulary(processed)
print('Vocabulário de tamanho', len(vocab))
print('Build vocab em %.2fs' % (time.time()-start))

start = time.time()
vectors = compute_tfidf(processed, vocab)
print('TF-IDF shape:', vectors.shape)
print('TF-IDF em %.2fs' % (time.time()-start))

start = time.time()
# gera features (usa X_train como exemplo)
if vectors.shape[0] > 1:
    X_train = vectors[:-1]
    new_features = generate_similarity_features(X_train)
    print('New features shape:', new_features.shape)
print('Feature gen em %.2fs' % (time.time()-start))

# Rodar um KNN simples para o último documento
if vectors.shape[0] > 1:
    pred = standard_knn(X_train, labels[:-1], vectors[-1:].reshape(1, -1), k=3)
    print('Predição KNN exemplo:', pred)

print('Debug pipeline concluída.')
