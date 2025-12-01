import re
import numpy as np

 # Pré-processamento simples sem NLTK: remoção de pontuação, minúsculas, tokenização.
def preprocess_texts(texts):
    processed = []
    for text in texts:
        text = re.sub(r'[^\w\s]', '', text.lower())  # Remove pontuação
        tokens = text.split()  # Tokenização simples
        processed.append(tokens)
    return processed

 # Construir vocabulário único
def build_vocabulary(processed_texts):
    vocab = set()
    for tokens in processed_texts:
        vocab.update(tokens)
    return list(vocab)