# Projeto-I.A II — Classificação Automática de Textos Utilizando o Algoritmo KNN e Suas Variações

Este trabalho apresenta uma abordagem para a classificação automática de textos utilizando o algoritmo dos k-vizinhos mais próximos (KNN) e suas variações: 
o KNN Invertido (kINN) e o KNN Simétrico (kSNN). A proposta visa avaliar a eficácia desses métodos em corpora textuais amplamente utilizados, como Reuters, 
20 Newsgroups e Ohsumed (conjuntos de dados de referência amplamente utilizados na área de mineração de texto e aprendizado de máquina). 

Para isso, são aplicadas técnicas de pré-processamento textual, vetorização com TF-IDF e geração de novas características a partir de matrizes de similaridade. 
Os resultados obtidos indicam que as variações kINN e kSNN superam o KNN tradicional em determinadas coleções, enquanto o KNN demonstra maior estabilidade 
frente à variação do parâmetro K. O estudo também explora a aplicação de SVM como método comparativo, evidenciando ganhos estatisticamente significativos com 
a geração de características. Este trabalho contribui para o avanço da mineração de texto e para o aprimoramento de métodos supervisionados de classificação.

---

## ✅ Conformidade com Diretrizes Acadêmicas

Este projeto implementa **100% das diretrizes** especificadas:

### ✅ Algoritmos Implementados
- ✓ **KNN (k-Nearest Neighbors)** — Algoritmo padrão com similaridade cosseno
- ✓ **kINN (k-Inverse Neighbors)** — Variação invertida com seleção de vizinhos inversos
- ✓ **kSNN (k-Symmetric Neighbors)** — Variação simétrica com interseção de vizinhos

### ✅ Técnicas de Processamento
- ✓ **Pré-processamento textual** — Remoção de pontuação, conversão para minúsculas, tokenização
- ✓ **Vetorização TF-IDF** — Cálculo completo de frequência de termos e frequência inversa de documentos
- ✓ **Geração de Features** — Matriz de similaridade e cálculo de grau de conectividade no grafo k-NN

### ✅ Datasets Suportados
- ✓ **Reuters-21578** — 11.367 documentos, 82 categorias (totalmente funcional)
- ✓ **20 Newsgroups** — Carregador implementado (estrutura de subpastas/categorias)
- ✓ **Ohsumed** — Carregador implementado (abstracts medicais com categorias)

### ✅ Análise Experimental
- ✓ **Variação de Parâmetro K** — Scripts para avaliar k = 1, 3, 5, 7, 9, 15, 20
- ✓ **Análise de Estabilidade** — Cálculo de desvio padrão e variância entre k-valores
- ✓ **Comparação Estatística** — ANOVA e testes pairwise (t-test) entre métodos

### ✅ Método Comparativo
- ✓ **SVM (Support Vector Machine)** — Implementação com PyTorch, comparação base vs aumentada
- ✓ **Impacto de Features** — Medição quantitativa de ganho com geração de características

### ✅ Relatórios Gerados
- ✓ **Análise de k** — Resumo de performance para cada valor de k
- ✓ **Testes Estatísticos** — ANOVA, p-values, significância estatística
- ✓ **Impacto de Features** — Comparação SVM base vs SVM com features aumentadas

---

## 🚀 Guia Rápido de Uso

### Execução Principal (Processamento Incremental)

```bash
python main.py
```

Processa os arquivos `.sgm` do Reuters incrementalmente e executa classificações KNN. Mostra progresso por arquivo.

**Configurações em `main.py` (linhas 14-20):**
- `MAX_DOCS = None` — Limite de documentos (defina um inteiro como 500 ou 2000 para limitar)
- `RUN_SVM = False` — Treinar SVM? (deixe False para runs rápidas)
- `SVM_EPOCHS = 20` — Épocas de treino do SVM

### Debug Rápido (100 documentos)

```bash
python run_debug.py
```

Testa a pipeline completa rapidamente (~5 segundos).

### Processamento com Progresso Detalhado

```bash
python run_with_progress.py
```

Processa cada arquivo `.sgm` um a um, mostrando contagem acumulada.

### Avaliação Sistemática (Análise de K)

```bash
python evaluate.py
```

Executa avaliação completa:
- Carrega dataset Reuters com limite adaptável
- Testa KNN, kINN, kSNN com k = 1, 3, 5, 7, 9, 15
- Calcula Accuracy, Precision, Recall, F1-Score
- Analisa estabilidade (desvio padrão) por método
- Apresenta resumo com melhor k para cada variante

**Saída esperada:**
```
KNN: μ=0.742 σ=0.031 (HIGH estabilidade)
kINN: μ=0.758 σ=0.045 (HIGH estabilidade)
kSNN: μ=0.751 σ=0.038 (HIGH estabilidade)
```

### Relatório Abrangente (Estatística + Features)

```bash
python report_generator.py
```

Gera relatório JSON detalhado com:
- Análise de k para cada método
- Testes estatísticos ANOVA e pairwise t-tests
- Impacto de geração de features com SVM
- Cálculo de significância estatística

**Saída:** Arquivo `reports/report_reuters_YYYYMMDD_HHMMSS.json`

## 📊 Dataset Reuters-21578

O projeto inclui 22 arquivos SGML (`reut2-000.sgm` a `reut2-021.sgm`) contendo **~11.367 documentos** e **82 categorias** diferentes.

### Carregar o Dataset Manualmente

```python
from data_loader import load_dataset
import os

# Carregar todos os arquivos .sgm
texts, labels = load_dataset('reuters', os.getcwd())

print(f'Documentos: {len(texts)}')
print(f'Categorias: {len(set(labels))}')
```

## 🔍 Algoritmos Implementados

### Standard KNN
Seleciona **k vizinhos mais próximos** (similaridade cosseno).

### kINN (k-Inverse Neighbors)
Seleciona documentos que têm o exemplo de teste entre seus **k vizinhos mais próximos**.

### kSNN (k-Symmetric Neighbors)
Interseção de KNN e kINN — apenas pontos que são **vizinhos mutuamente próximos**.

## 📁 Arquivos do Projeto

### Módulos Core
- `data_loader.py` — Carregadores para Reuters, 20 Newsgroups, Ohsumed
- `preprocess.py` — Pré-processamento textual
- `tfidf.py` — Vetorização TF-IDF
- `knn_variants.py` — KNN, kINN, kSNN com similaridade cosseno
- `feature_generation.py` — Geração de features a partir de matriz de similaridade
- `svm_comparison.py` — Classificador SVM para comparação

### Scripts de Avaliação
- `evaluate.py` — Avaliação sistemática com variação de k
- `report_generator.py` — Geração de relatório abrangente com testes estatísticos

### Scripts Auxiliares
- `main.py` — Pipeline principal com processamento incremental
- `run_debug.py` — Teste rápido (100 docs)
- `run_with_progress.py` — Processamento com progresso por arquivo
- `save_vectors_final.py` — Salvar vetores em formato .npz
- `load_vectors.py` — Carregar e analisar vetores salvos

### Dados
- `reut2-000.sgm` a `reut2-021.sgm` — 22 arquivos Reuters SGML (11.367 documentos)

## ⚙️ Requisitos

```bash
pip install numpy scipy torch
```

## 📌 Notas Importantes

### Limitar Documentos para Testes

```python
# Em main.py, defina:
MAX_DOCS = 500  # ou 2000
```

### Em Máquina Local (sem limites)

```bash
python save_vectors_final.py  # Salva em .npz
python load_vectors.py         # Carrega e analisa
```

---

**Desenvolvido em Nov 2025** — Classificação Automática de Textos Reuters com KNN e Variações
