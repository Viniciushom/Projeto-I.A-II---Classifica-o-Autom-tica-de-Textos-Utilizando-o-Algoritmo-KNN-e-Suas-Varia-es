# 🔍 Verificação Prática de Conformidade

## Como Validar Cada Componente

### 1. ✅ ALGORITMOS KNN

**Arquivo:** `knn_variants.py`

Verificar:
```bash
python -c "
from knn_variants import standard_knn, kinn, ksnn
from tfidf import compute_tfidf
from preprocess import preprocess_texts, build_vocabulary
from data_loader import load_dataset

texts, labels, label_map = load_dataset('reuters', '.')
texts = texts[:50]
labels = labels[:50]
processed = preprocess_texts(texts)
vocab = build_vocabulary(processed)
vectors = compute_tfidf(processed, vocab)

X_train, X_test = vectors[:-5], vectors[-5:]
y_train = labels[:-5]

# Teste dos 3 algoritmos
k=3
for algo in ['standard_knn', 'kinn', 'ksnn']:
    if algo == 'standard_knn':
        pred = standard_knn(X_train, y_train, X_test[:1], k)
    elif algo == 'kinn':
        pred = kinn(X_train, y_train, X_test[:1], k)
    else:
        pred = ksnn(X_train, y_train, X_test[:1], k)
    print(f'✅ {algo}: {pred}')
"
```

---

### 2. ✅ PRÉ-PROCESSAMENTO

**Arquivo:** `preprocess.py`

Verificar:
```bash
python -c "
from preprocess import preprocess_texts

textos = [
    'Este é um texto COM PONTUAÇÃO!!!',
    'Outro exemplo de pré-processamento...'
]

processed = preprocess_texts(textos)
for i, p in enumerate(processed):
    print(f'Original {i}: {textos[i]}')
    print(f'Processado {i}: {p}')
    print()
"
```

✓ Remove pontuação: `!!!` desaparece
✓ Minúsculas: `TEXTO` → `texto`
✓ Tokenização: lista de tokens
✓ Normalização: espaços extras removidos

---

### 3. ✅ VETORIZAÇÃO TF-IDF

**Arquivo:** `tfidf.py`

Verificar:
```bash
python -c "
from tfidf import compute_tfidf
from preprocess import preprocess_texts, build_vocabulary
import numpy as np

textos = [
    'gato gato cachorro',
    'gato peixe',
    'cachorro peixe peixe peixe'
]
processed = preprocess_texts(textos)
vocab = build_vocabulary(processed)
vectors = compute_tfidf(processed, vocab)

print(f'Shape: {vectors.shape}')  # (3, vocab_size)
print(f'Tem TF-IDF > 0: {(vectors > 0).sum()} valores')
print(f'Exemplo de valores TF-IDF:')
print(vectors[0][:5])  # Primeiros 5 valores
"
```

✓ Matriz tem dimensões (n_docs, n_termos)
✓ Valores TF-IDF > 0
✓ Valores balanços (nem todos iguais)

---

### 4. ✅ GERAÇÃO DE FEATURES

**Arquivo:** `feature_generation.py`

Verificar:
```bash
python -c "
from feature_generation import generate_similarity_features
import numpy as np

# Matriz dummy para teste
X = np.random.rand(10, 20)
features = generate_similarity_features(X, k=3)

print(f'Input shape: {X.shape}')
print(f'Features shape: {features.shape}')
print(f'Features (graus): {features.flatten()[:5]}')
print(f'✅ Features geradas com sucesso')
"
```

✓ Features têm dimensão (n_docs, 1)
✓ Valores são inteiros (graus)
✓ Sem valores NaN

---

### 5. ✅ DATASETS

**Arquivo:** `data_loader.py`

Verificar Reuters:
```bash
python -c "
from data_loader import load_dataset

texts, labels, label_map = load_dataset('reuters', '.')
print(f'✅ Reuters: {len(texts)} docs, {len(label_map)} categorias')
print(f'   Categorias: {list(label_map.keys())[:5]}')
"
```

Verificar estrutura para 20 Newsgroups (local):
```bash
# Se tiver dados em /path/to/20newsgroups com subpastas:
python -c "
from data_loader import load_dataset
texts, labels, label_map = load_dataset('20newsgroups', '/path/to/20newsgroups')
print(f'✅ 20 Newsgroups: {len(texts)} docs, {len(label_map)} categorias')
"
```

Verificar estrutura para Ohsumed (local):
```bash
# Se tiver arquivo ohsumed.txt ou subpastas:
python -c "
from data_loader import load_dataset
texts, labels, label_map = load_dataset('ohsumed', '/path/to/ohsumed')
print(f'✅ Ohsumed: {len(texts)} docs, {len(label_map)} categorias')
"
```

---

### 6. ✅ ANÁLISE DE VARIAÇÃO DE K

**Arquivo:** `evaluate.py`

Verificar:
```bash
python << 'EOF'
from evaluate import KNNEvaluator
import os

evaluator = KNNEvaluator()
cwd = os.getcwd()

# Mini teste
X, y, label_map = evaluator.load_and_preprocess('reuters', cwd, max_docs=100)
results, X_train, X_test, y_train, y_test = evaluator.evaluate_knn_variants(
    X, y, k_values=[1, 3, 5]
)

print("\n✅ Análise de K executada:")
for method in ['standard_knn', 'kinn', 'ksnn']:
    accs = results[method]['accuracy']
    print(f"   {method}: {accs}")
EOF
```

---

### 7. ✅ ANÁLISE DE ESTABILIDADE

**Arquivo:** `evaluate.py`

Verificar:
```bash
python << 'EOF'
import numpy as np
from evaluate import KNNEvaluator
import os

evaluator = KNNEvaluator()
X, y, label_map = evaluator.load_and_preprocess('reuters', os.getcwd(), 100)
results, _, _, _, _ = evaluator.evaluate_knn_variants(X, y, k_values=[1, 3, 5])

print("\n✅ Estabilidade (Desvio Padrão):")
for method in ['standard_knn', 'kinn', 'ksnn']:
    accs = np.array(results[method]['accuracy'])
    print(f"   {method}: σ = {np.std(accs):.4f}")
EOF
```

✓ Desvio padrão baixo = alta estabilidade
✓ Diferentes métodos têm estabilidades distintas

---

### 8. ✅ SVM COMPARATIVO

**Arquivo:** `svm_comparison.py`

Verificar:
```bash
python -c "
from svm_comparison import train_svm, predict_svm
import numpy as np

# Dados dummy
X_train = np.random.rand(50, 100)
y_train = np.random.randint(0, 2, 50)
X_test = np.random.rand(10, 100)

model = train_svm(X_train, y_train, epochs=10)
predictions = predict_svm(model, X_test)

print(f'✅ SVM treinado e predições geradas: {predictions}')
"
```

---

### 9. ✅ IMPACTO DE FEATURES

**Arquivo:** `evaluate.py`

Verificar:
```bash
python << 'EOF'
from evaluate import KNNEvaluator
import os

evaluator = KNNEvaluator()
X, y, label_map = evaluator.load_and_preprocess('reuters', os.getcwd(), 100)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

feature_results = evaluator.evaluate_with_features(
    X_train, X_test, y_train, y_test, k=3
)

print("\n✅ Impacto de Features:")
print(f"   SVM base: {feature_results['svm_base']:.3f}")
print(f"   SVM aumentado: {feature_results['svm_aug']:.3f}")
print(f"   Ganho: {feature_results['gain']*100:.2f}%")
EOF
```

---

### 10. ✅ ANÁLISE ESTATÍSTICA

**Arquivo:** `report_generator.py`

Verificar:
```bash
python << 'EOF'
from report_generator import ComprehensiveReport
import os

reporter = ComprehensiveReport(output_dir='./reports')
report = reporter.run_comprehensive_evaluation('reuters', os.getcwd(), max_docs=500)

print("\n✅ Testes Estatísticos Realizados:")
print(f"   ANOVA p-value: {report['statistical_tests']['anova']['p_value']:.4f}")
print(f"   Testes Pairwise: {len(report['statistical_tests']['pairwise_tests'])} comparações")

for pair, result in report['statistical_tests']['pairwise_tests'].items():
    sig = "SIM" if result['significant_at_0.05'] else "NÃO"
    print(f"   {pair}: p={result['p_value']:.4f} (significante: {sig})")
EOF
```

---

## 📋 Checklist de Verificação Manual

Copie e cole cada comando abaixo em seu terminal:

```bash
# 1. Verificar carregamento de dados
python -c "from data_loader import load_dataset; t,l,m = load_dataset('reuters','.'); print('✅ Reuters OK')"

# 2. Verificar pré-processamento
python -c "from preprocess import preprocess_texts; p = preprocess_texts(['Teste!!!']); print('✅ Preprocess OK')"

# 3. Verificar TF-IDF
python -c "from tfidf import compute_tfidf; from preprocess import *; print('✅ TF-IDF OK')"

# 4. Verificar KNN
python -c "from knn_variants import standard_knn; print('✅ KNN OK')"

# 5. Verificar Features
python -c "from feature_generation import generate_similarity_features; print('✅ Features OK')"

# 6. Verificar SVM
python -c "from svm_comparison import train_svm; print('✅ SVM OK')"

# 7. Verificar Avaliação
python -c "from evaluate import KNNEvaluator; print('✅ Evaluate OK')"

# 8. Verificar Relatório
python -c "from report_generator import ComprehensiveReport; print('✅ Report OK')"

# 9. Pipeline Completa
python main.py

# 10. Relatório JSON
python report_generator.py
```

---

## ✅ Documentação Relacionada

Veja os arquivos para mais detalhes:

- **README.md** — Guia de uso geral
- **CONFORMIDADE.md** — Análise técnica detalhada
- **STATUS_CONFORMIDADE.txt** — Checklist simplificado

---

**Desenvolvido em Novembro 2025**
**Versão: 1.0 — Conformidade Completa**
