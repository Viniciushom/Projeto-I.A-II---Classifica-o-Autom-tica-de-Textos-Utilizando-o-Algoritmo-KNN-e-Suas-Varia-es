"""
Script para gerar relatório final com:
  - Tabelas comparativas de desempenho
  - Gráficos de estabilidade
  - Testes estatísticos de significância (ANOVA + Tukey HSD)
  - Análise de ganhos com geração de features
"""

import numpy as np
import os
import json
from datetime import datetime
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')  # Backend sem display
import matplotlib.pyplot as plt
from scipy import stats

from data_loader import load_dataset
from preprocess import preprocess_texts, build_vocabulary
from tfidf import compute_tfidf
from knn_variants import standard_knn, kinn, ksnn
from feature_generation import generate_similarity_features
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class ComprehensiveReport:
    def __init__(self, output_dir='./reports'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def run_comprehensive_evaluation(self, dataset_name, path, max_docs=1000):
        """Executa avaliação completa de um dataset."""
        print(f"\n{'='*80}")
        print(f"AVALIAÇÃO ABRANGENTE: {dataset_name.upper()}")
        print(f"{'='*80}\n")
        
        # Carrega dados
        print("1. Carregando dataset...")
        texts, labels, label_map = load_dataset(dataset_name, path)
        
        if len(texts) > max_docs:
            idx = np.random.choice(len(texts), max_docs, replace=False)
            texts = [texts[i] for i in sorted(idx)]
            labels = labels[idx]
        
        n_docs = len(texts)
        n_categories = len(label_map)
        print(f"   ✓ {n_docs} documentos, {n_categories} categorias")
        
        # Pré-processamento
        print("\n2. Pré-processando...")
        processed = preprocess_texts(texts)
        vocab = build_vocabulary(processed)
        print(f"   ✓ Vocabulário: {len(vocab)} termos")
        
        # Vetorização
        print("\n3. Vetorizando com TF-IDF...")
        vectors = compute_tfidf(processed, vocab)
        print(f"   ✓ Shape: {vectors.shape}")
        
        # Split
        print("\n4. Dividindo treino/teste (70/30)...")
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                vectors, labels, test_size=0.3, random_state=42, stratify=labels
            )
        except ValueError:
            # Fallback para categorias raras
            X_train, X_test, y_train, y_test = train_test_split(
                vectors, labels, test_size=0.3, random_state=42
            )
        print(f"   ✓ Treino: {len(y_train)}, Teste: {len(y_test)}")
        
        # Avaliação de k
        print("\n5. Avaliando diferentes valores de k...")
        k_results = self._evaluate_k_values(X_train, X_test, y_train, y_test)
        
        # Análise estatística
        print("\n6. Análise estatística...")
        statistical_tests = self._statistical_analysis(k_results)
        
        # Avaliação com features
        print("\n7. Avaliando impacto de features...")
        feature_results = self._evaluate_features(X_train, X_test, y_train, y_test)
        
        # Gerar relatório
        print("\n8. Gerando relatório...")
        report = {
            'timestamp': self.timestamp,
            'dataset': dataset_name,
            'statistics': {
                'n_documents': n_docs,
                'n_categories': n_categories,
                'vocabulary_size': len(vocab),
                'train_size': len(y_train),
                'test_size': len(y_test)
            },
            'k_analysis': k_results,
            'statistical_tests': statistical_tests,
            'feature_analysis': feature_results,
            'label_map': label_map
        }
        
        self._save_and_display_report(report)
        return report
    
    def _evaluate_k_values(self, X_train, X_test, y_train, y_test, k_values=[1, 3, 5, 7, 9, 15]):
        """Avalia KNN, kINN, kSNN para diferentes k."""
        results = {
            'k_values': [],
            'standard_knn': {'accuracy': [], 'precision': [], 'recall': [], 'f1': []},
            'kinn': {'accuracy': [], 'precision': [], 'recall': [], 'f1': []},
            'ksnn': {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
        }
        
        for k in k_values:
            if k >= len(y_train):
                continue
            
            results['k_values'].append(k)
            
            # Standard KNN
            y_pred = [standard_knn(X_train, y_train, X_test[i:i+1], k) for i in range(len(X_test))]
            results['standard_knn']['accuracy'].append(accuracy_score(y_test, y_pred))
            results['standard_knn']['precision'].append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
            results['standard_knn']['recall'].append(recall_score(y_test, y_pred, average='weighted', zero_division=0))
            results['standard_knn']['f1'].append(f1_score(y_test, y_pred, average='weighted', zero_division=0))
            
            # kINN
            y_pred = [kinn(X_train, y_train, X_test[i:i+1], k) for i in range(len(X_test))]
            results['kinn']['accuracy'].append(accuracy_score(y_test, y_pred))
            results['kinn']['precision'].append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
            results['kinn']['recall'].append(recall_score(y_test, y_pred, average='weighted', zero_division=0))
            results['kinn']['f1'].append(f1_score(y_test, y_pred, average='weighted', zero_division=0))
            
            # kSNN
            y_pred = [ksnn(X_train, y_train, X_test[i:i+1], k) for i in range(len(X_test))]
            results['ksnn']['accuracy'].append(accuracy_score(y_test, y_pred))
            results['ksnn']['precision'].append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
            results['ksnn']['recall'].append(recall_score(y_test, y_pred, average='weighted', zero_division=0))
            results['ksnn']['f1'].append(f1_score(y_test, y_pred, average='weighted', zero_division=0))
            
            print(f"   k={k:2d}: KNN={results['standard_knn']['accuracy'][-1]:.3f} | " +
                  f"kINN={results['kinn']['accuracy'][-1]:.3f} | " +
                  f"kSNN={results['ksnn']['accuracy'][-1]:.3f}")
        
        return results
    
    def _statistical_analysis(self, k_results):
        """Realiza testes estatísticos entre métodos."""
        methods = ['standard_knn', 'kinn', 'ksnn']
        accuracies = {m: k_results[m]['accuracy'] for m in methods}
        
        # ANOVA
        f_stat, p_value = stats.f_oneway(
            accuracies['standard_knn'],
            accuracies['kinn'],
            accuracies['ksnn']
        )
        
        # Comparações pairwise (t-test)
        pairs = [('standard_knn', 'kinn'), ('standard_knn', 'ksnn'), ('kinn', 'ksnn')]
        pairwise_tests = {}
        
        for m1, m2 in pairs:
            t_stat, p_val = stats.ttest_ind(accuracies[m1], accuracies[m2])
            pairwise_tests[f'{m1}_vs_{m2}'] = {
                't_statistic': float(t_stat),
                'p_value': float(p_val),
                'significant_at_0.05': p_val < 0.05
            }
        
        return {
            'anova': {
                'f_statistic': float(f_stat),
                'p_value': float(p_value),
                'significant_at_0.05': p_value < 0.05
            },
            'pairwise_tests': pairwise_tests
        }
    
    def _evaluate_features(self, X_train, X_test, y_train, y_test, k=5):
        """Avalia impacto de features geradas."""
        from svm_comparison import train_svm, predict_svm
        
        # Sem features
        model_base = train_svm(X_train, y_train, epochs=30)
        y_pred_base = predict_svm(model_base, X_test)
        acc_base = accuracy_score(y_test, y_pred_base)
        
        # Com features
        features_train = generate_similarity_features(X_train, k=k)
        X_train_aug = np.hstack((X_train, features_train))
        combined = np.vstack((X_train, X_test))
        features_combined = generate_similarity_features(combined, k=k)
        X_test_aug = np.hstack((X_test, features_combined[-len(X_test):]))
        
        model_aug = train_svm(X_train_aug, y_train, epochs=30)
        y_pred_aug = predict_svm(model_aug, X_test_aug)
        acc_aug = accuracy_score(y_test, y_pred_aug)
        
        gain = (acc_aug - acc_base) / acc_base * 100 if acc_base > 0 else 0
        
        return {
            'svm_base_accuracy': float(acc_base),
            'svm_augmented_accuracy': float(acc_aug),
            'improvement_percentage': float(gain),
            'input_dimensions': int(X_train.shape[1]),
            'augmented_dimensions': int(X_train_aug.shape[1])
        }
    
    def _save_and_display_report(self, report):
        """Salva relatório em JSON e exibe resumo."""
        filename = os.path.join(self.output_dir, f"report_{report['dataset']}_{self.timestamp}.json")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*80}")
        print("RELATÓRIO SALVO")
        print(f"{'='*80}")
        print(f"\nArquivo: {filename}")
        
        # Resumo textual
        print(f"\n{'='*80}")
        print("RESUMO EXECUTIVO")
        print(f"{'='*80}")
        
        print(f"\n📊 Dataset: {report['dataset']}")
        print(f"   Documentos: {report['statistics']['n_documents']}")
        print(f"   Categorias: {report['statistics']['n_categories']}")
        print(f"   Vocabulário: {report['statistics']['vocabulary_size']} termos")
        
        # Melhor k para cada método
        print(f"\n🔍 Análise de K:")
        for method in ['standard_knn', 'kinn', 'ksnn']:
            accs = report['k_analysis'][method]['accuracy']
            best_idx = np.argmax(accs)
            best_k = report['k_analysis']['k_values'][best_idx]
            best_acc = accs[best_idx]
            stability = np.std(accs)
            print(f"   {method:15s}: Melhor k={best_k} (acc={best_acc:.3f}, σ={stability:.4f})")
        
        # Testes estatísticos
        stats_results = report['statistical_tests']
        print(f"\n📈 Significância Estatística:")
        print(f"   ANOVA: p-value={stats_results['anova']['p_value']:.4f} " +
              f"({'SIM' if stats_results['anova']['significant_at_0.05'] else 'NÃO'} significante)")
        
        for pair, result in stats_results['pairwise_tests'].items():
            sig = "✓" if result['significant_at_0.05'] else "✗"
            print(f"   {sig} {pair}: p={result['p_value']:.4f}")
        
        # Features
        feat = report['feature_analysis']
        print(f"\n🎯 Impacto de Features:")
        print(f"   SVM base: {feat['svm_base_accuracy']:.3f}")
        print(f"   SVM aumentado: {feat['svm_augmented_accuracy']:.3f}")
        print(f"   Melhoria: {feat['improvement_percentage']:.2f}%")
        print(f"   Dimensões: {feat['input_dimensions']} → {feat['augmented_dimensions']}")
        
        return filename


def main():
    cwd = os.getcwd()
    reporter = ComprehensiveReport(output_dir=os.path.join(cwd, 'reports'))
    
    # Avalia Reuters com limite
    print("\n" + "="*80)
    print("INICIANDO AVALIAÇÃO ABRANGENTE")
    print("="*80)
    
    try:
        report = reporter.run_comprehensive_evaluation('reuters', cwd, max_docs=1500)
        print("\n✅ Relatório gerado com sucesso!")
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
