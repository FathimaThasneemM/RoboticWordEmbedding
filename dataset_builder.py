import re
import csv
import numpy as np
from collections import Counter

corpus = """
A mobile robot navigates a warehouse by combining lidar scans with odometry to build  
a map. The controller plans collision-free paths, adjusts speed near obstacles, and  
executes docking for charging. Manipulators pick items using grasp planning and force  
feedback for stable handling. Preventive maintenance checks motors, encoders, and  
calibration to maintain accuracy.
"""

def preprocess_text(text):
    # Case normalization and punctuation removal
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    return tokens

def build_vocab(tokens):
    counts = Counter(tokens)
    vocab = {word: (i, counts[word]) for i, word in enumerate(sorted(counts.keys()))}
    # Save to vocab.txt: word, ID, frequency
    with open('vocab.txt', 'w') as f:
        for word, (idx, freq) in vocab.items():
            f.write(f"{word} {idx} {freq}\n")
    return vocab

def generate_datasets(tokens, vocab, window_size=4):
    word_to_id = {w: i for w, (i, f) in vocab.items()}
    token_ids = [word_to_id[t] for t in tokens]
    
    cbow_data = []
    skipgram_data = []

    for i in range(window_size, len(token_ids) - window_size):
        target = token_ids[i]
        context = token_ids[i-window_size:i] + token_ids[i+1:i+window_size+1]
        
        # CBOW: [context_words] -> target
        cbow_data.append(context + [target])
        
        # Skip-gram: target -> context_word (one pair per context word)
        for ctx in context:
            skipgram_data.append([target, ctx])

    # Save CBOW
    with open('cbow_dataset.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(cbow_data)
        
    # Save Skip-gram
    with open('skipgram_dataset.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(skipgram_data)

tokens = preprocess_text(corpus)
vocab = build_vocab(tokens)
generate_datasets(tokens, vocab)
print("Preprocessing complete. vocab.txt and datasets generated.")