import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def evaluate(csv_file, result_label):
    # Load vocab and embeddings
    words = []
    with open('vocab.txt', 'r') as f:
        for line in f:
            words.append(line.split()[0])
            
    embeddings = pd.read_csv(csv_file).values
    query_words = ['robot', 'warehouse', 'lidar', 'controller', 'maintenance']
    
    with open('similarity_results.txt', 'a') as f:
        f.write(f"\n--- Results for {result_label} ---\n")
        for qw in query_words:
            if qw not in words: continue
            
            idx = words.index(qw)
            vec = embeddings[idx].reshape(1, -1)
            
            # Calculate similarity against all other words
            sims = cosine_similarity(vec, embeddings)[0]
            
            # Get Top 5 (excluding the word itself)
            nearest_indices = sims.argsort()[-6:-1][::-1]
            
            f.write(f"Query: {qw}\n")
            for i in nearest_indices:
                f.write(f"  {words[i]}: {sims[i]:.4f}\n")
    print(f"Similarity results for {result_label} saved.")

if __name__ == "__main__":
    open('similarity_results.txt', 'w').close() # Clear file
    evaluate('embeddings_cbow.csv', 'CBOW')
    evaluate('embeddings_skipgram.csv', 'Skip-gram')