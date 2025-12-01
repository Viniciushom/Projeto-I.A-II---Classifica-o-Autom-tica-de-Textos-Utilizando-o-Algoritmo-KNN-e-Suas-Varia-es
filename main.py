from data_loader import load_dataset, _parse_reuters_sgml_file
from preprocess import preprocess_texts, build_vocabulary
from tfidf import compute_tfidf
from knn_variants import standard_knn, kinn, ksnn
from feature_generation import generate_similarity_features
from svm_comparison import train_svm, predict_svm
import numpy as np
import os
from collections import Counter


# Configurações interativas (ajuste antes de rodar)
# Se True, os arquivos .sgm são lidos um a um e mostra-se progresso.
PROCESS_INCREMENTALLY = True
# Limite de documentos (None para todos)
MAX_DOCS = None
# Treinar SVM por padrão? Em runs interativas deixe False (evita longos treinos)
RUN_SVM = False
# Se treinar SVM, use este número de épocas reduzido por padrão
SVM_EPOCHS = 20


def main(dataset_name='reuters', custom_path=None):
    """
    Pipeline principal de classificação.
    
    Args:
        dataset_name: 'reuters', '20newsgroups', ou 'ohsumed'
        custom_path: caminho customizado (None = diretório atual)
    """
    cwd = custom_path or os.getcwd()
    
    texts = []
    labels = []
    label_map = {}

    if PROCESS_INCREMENTALLY and dataset_name.lower() in ('reuters',):
        files = [f for f in sorted(os.listdir(cwd)) if f.lower().endswith('.sgm') or f.lower().startswith('reut')]
        if not files:
            raise RuntimeError('Nenhum arquivo .sgm encontrado no diretório atual')

        print(f'Processando {len(files)} arquivos .sgm incrementalmente...')
        for i, fname in enumerate(files, 1):
            fpath = os.path.join(cwd, fname)
            _parse_reuters_sgml_file(fpath, label_map, texts, labels)
            print(f'[{i}/{len(files)}] {fname}: total docs = {len(texts)}')
            if MAX_DOCS and len(texts) >= MAX_DOCS:
                texts = texts[:MAX_DOCS]
                labels = labels[:MAX_DOCS]
                print(f'Reached MAX_DOCS={MAX_DOCS}, stopping incremental load')
                break
    else:
        texts, labels, label_map = load_dataset(dataset_name, cwd)

    print('\nResumo do carregamento:')
    print('Documentos carregados:', len(texts))
    
    # Criar mapa reverso de índices para nomes de categorias
    if label_map:
        index_to_category = {v: k for k, v in label_map.items()}
        print(f'Categorias encontradas: {len(label_map)}')
        print('\nLista de todas as categorias:')
        for idx in sorted(index_to_category.keys()):
            count = (np.array(labels) == idx).sum()
            print(f'  [{idx:2d}] {index_to_category[idx]:20s} — {count:5d} documentos')
    else:
        print('Categorias encontradas:', len(set(labels)))
    
    if len(labels) > 0:
        print('\nTop 5 categorias (por frequência):')
        for label, count in Counter(labels).most_common(5):
            cat_name = index_to_category.get(label, f'cat_{label}') if label_map else f'cat_{label}'
            print(f'  {cat_name:20s} — {count:5d} documentos')

    # Pré-processamento e vetorização (pode demorar para todo o dataset)
    processed = preprocess_texts(texts)
    vocab = build_vocabulary(processed)
    print(f'Vocabulário de tamanho {len(vocab)}')
    vectors = compute_tfidf(processed, vocab)
    print('TF-IDF calculado, shape:', vectors.shape)

    # Dividir train/test (exemplo simples: último documento como teste)
    if vectors.shape[0] < 2:
        raise RuntimeError('Não há dados suficientes para treinar/testar')

    X_train, X_test = vectors[:-1], vectors[-1:]
    y_train, y_test = np.array(labels[:-1]), np.array(labels[-1:])

    # Gerar features adicionais (exemplo)
    try:
        new_features = generate_similarity_features(X_train)
        X_train_aug = np.hstack((X_train, new_features))
        X_test_aug = np.hstack((X_test, generate_similarity_features(np.vstack((X_train, X_test)))[-1:]))
    except Exception:
        print('Falha ao gerar/concatenar features adicionais — usando matrizes originais')
        X_train_aug = X_train
        X_test_aug = X_test

    # Classificações rápidas com KNN
    k = 3
    print('\nExemplo de predições com KNN (k=%d):' % k)
    try:
        print('KNN:', standard_knn(X_train, y_train, X_test, k))
        print('kINN:', kinn(X_train, y_train, X_test, k))
        print('kSNN:', ksnn(X_train, y_train, X_test, k))
    except Exception as e:
        print('Erro ao rodar classificadores KNN:', e)

    # Treino SVM opcional
    if RUN_SVM:
        print('\nTreinando SVM (padrão reduzido para interatividade)...')
        model = train_svm(X_train, y_train, epochs=SVM_EPOCHS)
        print('SVM (base):', predict_svm(model, X_test))
        model_aug = train_svm(X_train_aug, y_train, epochs=SVM_EPOCHS)
        print('SVM (aug):', predict_svm(model_aug, X_test_aug))
    else:
        print('\nTreino SVM pulado (RUN_SVM=False). Para treinar, habilite RUN_SVM no início do arquivo.')


if __name__ == '__main__':
    main()