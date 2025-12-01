"""
save_vectors_final.py: Versão corrigida que salva texto progressivamente
"""
import os
import json
import pickle
import numpy as np
from scipy.sparse import csr_matrix, save_npz
from data_loader import _parse_reuters_sgml_file
from preprocess import preprocess_texts, build_vocabulary
from tfidf import compute_tfidf

cwd = os.getcwd()
texts_file = 'reuters_texts.pkl'
labels_file = 'reuters_labels.pkl'
vocab_file = 'reuters_vocab.pkl'
checkpoint_file = 'parse_checkpoint.json'

files = [f for f in sorted(os.listdir(cwd)) if f.lower().endswith('.sgm') or f.lower().startswith('reut')]
if not files:
    raise RuntimeError('Nenhum arquivo .sgm encontrado')

# Carregar dados anteriores se existirem
if os.path.exists(texts_file) and os.path.exists(checkpoint_file):
    print('Carregando dados e checkpoint anteriores...')
    with open(texts_file, 'rb') as f:
        texts = pickle.load(f)
    with open(labels_file, 'rb') as f:
        labels = pickle.load(f)
    with open(checkpoint_file) as f:
        checkpoint = json.load(f)
    start_idx = checkpoint['files_processed']
    label_map = checkpoint['label_map']
    label_map = {int(k) if k.isdigit() else k: v for k, v in label_map.items()}
else:
    texts = []
    labels = []
    label_map = {}
    start_idx = 0
    checkpoint = {'files_processed': 0}

print(f'Processando {len(files)} arquivos (começando do {start_idx})...')

# Processar cada arquivo
for i, fname in enumerate(files[start_idx:], start=start_idx):
    fpath = os.path.join(cwd, fname)
    before = len(texts)
    _parse_reuters_sgml_file(fpath, label_map, texts, labels)
    after = len(texts)
    
    print(f'[{i+1}/{len(files)}] {fname}: +{after-before} docs')
    
    # Salvar progresso após cada arquivo
    with open(texts_file, 'wb') as f:
        pickle.dump(texts, f)
    with open(labels_file, 'wb') as f:
        pickle.dump(labels, f)
    
    checkpoint['files_processed'] = i + 1
    checkpoint['total_docs'] = len(texts)
    checkpoint['label_map'] = {str(k): v for k, v in label_map.items()}
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f)

print(f'\n✓ Parse concluído: {len(texts)} documentos, {len(label_map)} categorias')

# Agora vectorizar
print('\nPré-processando e construindo vocabulário...')
processed = preprocess_texts(texts)
vocab = build_vocabulary(processed)
print(f'Vocabulário: {len(vocab)} termos')

print('Computando TF-IDF...')
vectors = compute_tfidf(processed, vocab)
print(f'TF-IDF shape: {vectors.shape}')

# Salvar em sparse format
vectors_sparse = csr_matrix(vectors)
print(f'Salvando vetores ({vectors_sparse.data.nbytes / (1024**2):.2f} MB)...')
save_npz('reuters_vectors.npz', vectors_sparse)

np.savez(
    'reuters_metadata.npz',
    labels=np.array(labels),
    label_map_keys=np.array(list(label_map.keys())),
    label_map_values=np.array(list(label_map.values())),
    vocab=np.array(vocab)
)

print('\n✓ Vetores salvos em reuters_vectors.npz')
print('✓ Metadados salvos em reuters_metadata.npz')
print('\nPronto para usar load_vectors.py!')
