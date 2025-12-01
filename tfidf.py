import numpy as np

def compute_tfidf(processed_texts, vocab):
    n_docs = len(processed_texts)
    n_terms = len(vocab)
    term_to_idx = {term: idx for idx, term in enumerate(vocab)}
    
    # Matriz TF (term frequency)
    tf = np.zeros((n_docs, n_terms))
    for i, tokens in enumerate(processed_texts):
        token_counts = np.bincount([term_to_idx[t] for t in tokens if t in term_to_idx])
        tf[i, :len(token_counts)] = token_counts
    
    # DF (document frequency)
    df = np.sum(tf > 0, axis=0)
    
    # IDF = log(N / DF)
    idf = np.log(n_docs / (df + 1))  # Suavização
    
    # TF-IDF
    tfidf = tf * idf
    return tfidf