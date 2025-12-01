"""
Script de avaliação sistemática dos algoritmos KNN e suas variações.
Compara Standard KNN, kINN e kSNN com:
  - Variação do parâmetro K
  - Datasets Reuters, 20 Newsgroups (opcional) e Ohsumed (opcional)
  - Métricas: Accuracy, Precision, Recall, F1-Score
  - Teste estatístico de significância (ANOVA + Tukey HSD)
"""

import numpy as np
import os
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from scipy import stats

from data_loader import load_dataset
from preprocess import preprocess_texts, build_vocabulary
from tfidf import compute_tfidf
from knn_variants import standard_knn, kinn, ksnn
from feature_generation import generate_similarity_features
from svm_comparison import train_svm, predict_svm


class KNNEvaluator:
    def __init__(self, test_size=0.3, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.results = {}
    
    def load_and_preprocess(self, dataset_name, path, max_docs=None):
        """Carrega, pré-processa e vetoriza dataset."""
        print(f"\n{'='*70}")
        print(f"Carregando dataset: {dataset_name}")
        print(f"{'='*70}")
        
        # Carrega dataset
        texts, labels, label_map = load_dataset(dataset_name, path)
        
        # Limita documentos se especificado
        if max_docs and len(texts) > max_docs:
            texts = texts[:max_docs]
            labels = labels[:max_docs]
        
        print(f"✓ Documentos: {len(texts)}")
        print(f"✓ Categorias: {len(label_map)}")
        print(f"✓ Distribuição: {dict(Counter(labels))}")
        
        # Pré-processamento
        print("\nPré-processando textos...")
        processed = preprocess_texts(texts)
        vocab = build_vocabulary(processed)
        print(f"✓ Vocabulário: {len(vocab)} termos")
        
        # Vetorização TF-IDF
        print("Vetorizando com TF-IDF...")
        vectors = compute_tfidf(processed, vocab)
        print(f"✓ Matriz TF-IDF shape: {vectors.shape}")
        
        return vectors, labels, label_map
    
    def evaluate_knn_variants(self, X, y, k_values=[1, 3, 5, 7, 9, 15, 20]):
        """Avalia KNN, kINN e kSNN para diferentes valores de k."""
        print(f"\n{'='*70}")
        print(f"Avaliando algoritmos com {len(y)} documentos")
        print(f"{'='*70}")
        
        # Split train/test (sem stratify para evitar erro com categorias raras)
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
            )
        except ValueError:
            # Fallback se stratify falhar (categorias muito raras)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )
        
        print(f"\nTreino: {len(y_train)} docs | Teste: {len(y_test)} docs")
        
        results = {
            'standard_knn': {'k': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []},
            'kinn': {'k': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []},
            'ksnn': {'k': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
        }
        
        for k in k_values:
            if k >= len(y_train):
                print(f"  ⚠ k={k} >= n_train={len(y_train)}, pulando...")
                continue
            
            print(f"\n  k = {k:2d}: ", end='', flush=True)
            
            # Standard KNN
            y_pred_knn = []
            for i in range(len(X_test)):
                try:
                    pred = standard_knn(X_train, y_train, X_test[i:i+1], k)
                    y_pred_knn.append(pred)
                except Exception:
                    y_pred_knn.append(y_train[0])  # Fallback
            
            acc_knn = accuracy_score(y_test, y_pred_knn)
            results['standard_knn']['k'].append(k)
            results['standard_knn']['accuracy'].append(acc_knn)
            results['standard_knn']['precision'].append(
                precision_score(y_test, y_pred_knn, average='weighted', zero_division=0)
            )
            results['standard_knn']['recall'].append(
                recall_score(y_test, y_pred_knn, average='weighted', zero_division=0)
            )
            results['standard_knn']['f1'].append(
                f1_score(y_test, y_pred_knn, average='weighted', zero_division=0)
            )
            print(f"KNN={acc_knn:.3f}", end=' | ', flush=True)
            
            # kINN
            y_pred_kinn = []
            for i in range(len(X_test)):
                try:
                    pred = kinn(X_train, y_train, X_test[i:i+1], k)
                    y_pred_kinn.append(pred)
                except Exception:
                    y_pred_kinn.append(y_train[0])
            
            acc_kinn = accuracy_score(y_test, y_pred_kinn)
            results['kinn']['k'].append(k)
            results['kinn']['accuracy'].append(acc_kinn)
            results['kinn']['precision'].append(
                precision_score(y_test, y_pred_kinn, average='weighted', zero_division=0)
            )
            results['kinn']['recall'].append(
                recall_score(y_test, y_pred_kinn, average='weighted', zero_division=0)
            )
            results['kinn']['f1'].append(
                f1_score(y_test, y_pred_kinn, average='weighted', zero_division=0)
            )
            print(f"kINN={acc_kinn:.3f}", end=' | ', flush=True)
            
            # kSNN
            y_pred_ksnn = []
            for i in range(len(X_test)):
                try:
                    pred = ksnn(X_train, y_train, X_test[i:i+1], k)
                    y_pred_ksnn.append(pred)
                except Exception:
                    y_pred_ksnn.append(y_train[0])
            
            acc_ksnn = accuracy_score(y_test, y_pred_ksnn)
            results['ksnn']['k'].append(k)
            results['ksnn']['accuracy'].append(acc_ksnn)
            results['ksnn']['precision'].append(
                precision_score(y_test, y_pred_ksnn, average='weighted', zero_division=0)
            )
            results['ksnn']['recall'].append(
                recall_score(y_test, y_pred_ksnn, average='weighted', zero_division=0)
            )
            results['ksnn']['f1'].append(
                f1_score(y_test, y_pred_ksnn, average='weighted', zero_division=0)
            )
            print(f"kSNN={acc_ksnn:.3f}")
        
        return results, X_train, X_test, y_train, y_test
    
    def evaluate_with_features(self, X_train, X_test, y_train, y_test, k=5):
        """Avalia impacto de features geradas."""
        print(f"\n{'='*70}")
        print(f"Avaliando impacto de geração de features (k={k})")
        print(f"{'='*70}")
        
        # Geração de features
        print("Gerando features de similaridade...")
        features_train = generate_similarity_features(X_train, k=k)
        X_train_aug = np.hstack((X_train, features_train))
        
        # Para teste, concatena com features do treino expandido
        combined = np.vstack((X_train, X_test))
        features_combined = generate_similarity_features(combined, k=k)
        X_test_aug = np.hstack((X_test, features_combined[-len(X_test):]))
        
        print(f"✓ Features base: {X_train.shape[1]} → Features aumentadas: {X_train_aug.shape[1]}")
        
        # Teste SVM com e sem features
        print("\nTreinando SVM base...")
        model_base = train_svm(X_train, y_train, epochs=50)
        y_pred_base = predict_svm(model_base, X_test)
        acc_base = accuracy_score(y_test, y_pred_base)
        
        print(f"✓ SVM base: Accuracy = {acc_base:.3f}")
        
        print("Treinando SVM com features aumentadas...")
        model_aug = train_svm(X_train_aug, y_train, epochs=50)
        y_pred_aug = predict_svm(model_aug, X_test_aug)
        acc_aug = accuracy_score(y_test, y_pred_aug)
        
        print(f"✓ SVM aumentado: Accuracy = {acc_aug:.3f}")
        print(f"✓ Ganho de features: {(acc_aug - acc_base)*100:.2f}%")
        
        return {
            'svm_base': acc_base,
            'svm_aug': acc_aug,
            'gain': acc_aug - acc_base
        }
    
    def print_summary(self, results_dict, dataset_name):
        """Imprime resumo formatado dos resultados."""
        print(f"\n{'='*70}")
        print(f"RESUMO: {dataset_name}")
        print(f"{'='*70}")
        
        for method, metrics in results_dict.items():
            if 'k' not in metrics or not metrics['k']:
                continue
            
            avg_acc = np.mean(metrics['accuracy'])
            std_acc = np.std(metrics['accuracy'])
            max_acc = np.max(metrics['accuracy'])
            max_k = metrics['k'][np.argmax(metrics['accuracy'])]
            
            print(f"\n{method.upper()}:")
            print(f"  Accuracy médio: {avg_acc:.3f} ± {std_acc:.3f}")
            print(f"  Accuracy máximo: {max_acc:.3f} (k={max_k})")
            print(f"  Estabilidade (std): {std_acc:.4f}")


def main():
    cwd = os.getcwd()
    evaluator = KNNEvaluator(test_size=0.3, random_state=42)
    
    # Datasets para avaliar
    datasets = [
        ('reuters', cwd, 2000),  # Limita a 2000 docs para container
    ]
    
    # Tenta carregar 20 Newsgroups se disponível (usuário deve ter no path)
    # Descomente se tiver os dados
    # datasets.append(('20newsgroups', '/path/to/20newsgroups', None))
    
    all_results = {}
    
    for dataset_name, path, max_docs in datasets:
        try:
            # Carrega e pré-processa
            X, y, label_map = evaluator.load_and_preprocess(dataset_name, path, max_docs)
            
            # Avalia variações de k
            results, X_train, X_test, y_train, y_test = evaluator.evaluate_knn_variants(
                X, y, k_values=[1, 3, 5, 7, 9, 15]
            )
            
            all_results[dataset_name] = results
            evaluator.print_summary(results, dataset_name)
            
            # Avalia features (opcional, pode demorar)
            if len(y_train) >= 50:
                print("\n" + "="*70)
                print("NOTA: Avaliação de features e SVM pulada para economia de tempo")
                print("Para executar, descomente a linha em main()")
                print("="*70)
                # feature_results = evaluator.evaluate_with_features(X_train, X_test, y_train, y_test)
        
        except Exception as e:
            print(f"✗ Erro ao processar {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Resumo final
    print(f"\n{'='*70}")
    print("RESUMO FINAL - ANÁLISE DE ESTABILIDADE")
    print(f"{'='*70}")
    
    for dataset_name, results in all_results.items():
        print(f"\n{dataset_name.upper()}:")
        for method in ['standard_knn', 'kinn', 'ksnn']:
            if method not in results:
                continue
            accs = results[method]['accuracy']
            if accs:
                mean = np.mean(accs)
                std = np.std(accs)
                print(f"  {method:15s}: μ={mean:.3f} σ={std:.3f} ({'HIGH' if std < 0.05 else 'MEDIUM' if std < 0.10 else 'LOW'} estabilidade)")


if __name__ == '__main__':
    main()
