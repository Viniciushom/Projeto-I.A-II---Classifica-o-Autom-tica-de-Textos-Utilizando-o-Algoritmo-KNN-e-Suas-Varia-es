import os
import re
import os
import re
import html
import numpy as np


def _strip_tags(text):
    # Remove quaisquer tags HTML/SGML simples
    return re.sub(r"<[^>]+>", " ", text)


def _parse_reuters_sgml_file(file_path, label_map, texts, labels):
    """Parseia um único arquivo reut2-*.sgm e adiciona textos e labels às listas fornecidas.

    Regras aplicadas:
    - Extrai blocos <REUTERS>...</REUTERS>
    - Dentro de cada REUTERS extrai <TOPICS> com múltiplos <D>Topic</D>
    - Usa o primeiro tópico como rótulo; ignora documentos sem tópico
    - Extrai <TEXT> e concatena TITLE + BODY (se presentes), removendo tags internas
    """
    with open(file_path, 'r', encoding='latin-1', errors='ignore') as f:
        content = f.read()

    # Encontra cada bloco REUTERS
    reuters_blocks = re.findall(r"<REUTERS\b[^>]*>(.*?)</REUTERS>", content, flags=re.DOTALL | re.IGNORECASE)
    for block in reuters_blocks:
        # Extrai tópicos
        topics_block = re.search(r"<TOPICS>(.*?)</TOPICS>", block, flags=re.DOTALL | re.IGNORECASE)
        topics = []
        if topics_block:
            topics = re.findall(r"<D>(.*?)</D>", topics_block.group(1), flags=re.DOTALL | re.IGNORECASE)
            topics = [t.strip() for t in topics if t.strip()]

        if not topics:
            # Ignora documentos sem tópico (não rotulados)
            continue

        # Usa o primeiro tópico como rótulo principal
        topic = topics[0]
        if topic not in label_map:
            label_map[topic] = len(label_map)

        # Extrai o texto (TITLE e BODY dentro de <TEXT>)
        text_block = re.search(r"<TEXT\b[^>]*>(.*?)</TEXT>", block, flags=re.DOTALL | re.IGNORECASE)
        text = ""
        if text_block:
            inner = text_block.group(1)
            # Pode haver <TITLE> e/ou <BODY>
            title = ""
            body = ""
            title_m = re.search(r"<TITLE>(.*?)</TITLE>", inner, flags=re.DOTALL | re.IGNORECASE)
            if title_m:
                title = title_m.group(1)
            body_m = re.search(r"<BODY>(.*?)</BODY>", inner, flags=re.DOTALL | re.IGNORECASE)
            if body_m:
                body = body_m.group(1)
            # Se não houver BODY/TITLE, usamos todo o inner (alguns arquivos têm texto direto)
            if not (title or body):
                text = inner
            else:
                text = (title + "\n" + body).strip()

            # Remove tags e unescape de entidades
            text = _strip_tags(text)
            text = html.unescape(text)
            # Normaliza espaços
            text = re.sub(r"\s+", " ", text).strip()

        if not text:
            # pula se não há conteúdo textual
            continue

        texts.append(text)
        labels.append(label_map[topic])


def load_dataset(dataset_name, path):
    """
    Carrega datasets em formato padronizado: (texts, labels, label_map).
    Se chamado com 2 argumentos para compatibilidade com código antigo,
    retorna (texts, labels, label_map).
    
    Suporta: reuters, 20newsgroups, ohsumed
    """
    if dataset_name.lower() in ('20newsgroups', '20news'):
        return _load_20newsgroups(path)
    
    elif dataset_name.lower() in ('reuters', 'reuters21578'):
        return _load_reuters(path)
    
    elif dataset_name.lower() in ('ohsumed',):
        return _load_ohsumed(path)
    
    else:
        raise ValueError(f"Dataset '{dataset_name}' não suportado. Use: reuters, 20newsgroups, ohsumed")


def _load_reuters(path):
    """Carrega Reuters-21578 a partir de arquivos SGML."""
    texts = []
    labels = []
    label_map = {}

    # Se path for diretório, procura por arquivos reut*.sgm
    if os.path.isdir(path):
        for fname in sorted(os.listdir(path)):
            if fname.lower().endswith('.sgm') or fname.lower().startswith('reut'):
                fpath = os.path.join(path, fname)
                _parse_reuters_sgml_file(fpath, label_map, texts, labels)
    else:
        # path pode ser um arquivo único
        _parse_reuters_sgml_file(path, label_map, texts, labels)

    return texts, np.array(labels), label_map


def _load_20newsgroups(path):
    """
    Carrega 20 Newsgroups.
    Esperado: path com subpastas de categorias contendo arquivos .txt
    Exemplo: path/alt.atheism/*.txt, path/comp.graphics/*.txt, etc.
    """
    texts = []
    labels = []
    label_map = {}
    
    if not os.path.isdir(path):
        raise ValueError(f"20 Newsgroups: path '{path}' não é um diretório válido")
    
    # Procura por subpastas (categorias)
    for category_dir in sorted(os.listdir(path)):
        category_path = os.path.join(path, category_dir)
        
        # Ignora se não for diretório ou se for diretório especial
        if not os.path.isdir(category_path) or category_dir.startswith('.'):
            continue
        
        # Mapeia categoria para índice
        if category_dir not in label_map:
            label_map[category_dir] = len(label_map)
        
        category_idx = label_map[category_dir]
        
        # Lê todos os arquivos .txt na categoria
        for file in sorted(os.listdir(category_path)):
            file_path = os.path.join(category_path, file)
            
            if not os.path.isfile(file_path):
                continue
            
            try:
                with open(file_path, 'r', encoding='latin-1', errors='ignore') as f:
                    text = f.read().strip()
                    if text:  # Ignora arquivos vazios
                        texts.append(text)
                        labels.append(category_idx)
            except Exception:
                pass  # Ignora erros de leitura
    
    if not texts:
        raise ValueError("20 Newsgroups: nenhum arquivo encontrado no path")
    
    return texts, np.array(labels), label_map


def _load_ohsumed(path):
    """
    Carrega Ohsumed dataset.
    Esperado: arquivo 'ohsumed.txt' com formato:
    categorie|abstract
    ou subpastas por categoria com abstracts
    """
    texts = []
    labels = []
    label_map = {}
    
    if not os.path.exists(path):
        raise ValueError(f"Ohsumed: path '{path}' não encontrado")
    
    # Tenta carregar arquivo único ohsumed.txt
    if os.path.isfile(path):
        try:
            with open(path, 'r', encoding='latin-1', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if not line or '|' not in line:
                        continue
                    
                    parts = line.split('|', 1)
                    if len(parts) != 2:
                        continue
                    
                    category, text = parts
                    category = category.strip()
                    text = text.strip()
                    
                    if not text or not category:
                        continue
                    
                    if category not in label_map:
                        label_map[category] = len(label_map)
                    
                    texts.append(text)
                    labels.append(label_map[category])
        except Exception as e:
            raise ValueError(f"Ohsumed: erro ao ler arquivo {path}: {e}")
    
    # Tenta carregar de subpastas (alternativa)
    elif os.path.isdir(path):
        for category_dir in sorted(os.listdir(path)):
            category_path = os.path.join(path, category_dir)
            
            if not os.path.isdir(category_path) or category_dir.startswith('.'):
                continue
            
            if category_dir not in label_map:
                label_map[category_dir] = len(label_map)
            
            category_idx = label_map[category_dir]
            
            for file in sorted(os.listdir(category_path)):
                file_path = os.path.join(category_path, file)
                
                if not os.path.isfile(file_path):
                    continue
                
                try:
                    with open(file_path, 'r', encoding='latin-1', errors='ignore') as f:
                        text = f.read().strip()
                        if text:
                            texts.append(text)
                            labels.append(category_idx)
                except Exception:
                    pass
    
    if not texts:
        raise ValueError("Ohsumed: nenhum dado encontrado")
    
    return texts, np.array(labels), label_map