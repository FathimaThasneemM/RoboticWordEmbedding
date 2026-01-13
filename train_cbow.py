import pandas as pd
import numpy as np
import argparse
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

def train_cbow(epochs, lr):
    # Load vocab size
    vocab = {}
    with open('vocab.txt', 'r') as f:
        for line in f:
            parts = line.split()
            vocab[parts[0]] = int(parts[1])
    V = len(vocab)
    D = 10

    # Load data
    df = pd.read_csv('cbow_dataset.csv', header=None)
    X_raw = df.iloc[:, :-1].values # Context IDs
    y_raw = df.iloc[:, -1].values  # Target ID

    # Convert to One-Hot
    # X for CBOW is the sum/mean of one-hot context vectors
    X = np.zeros((len(X_raw), V))
    for i, row in enumerate(X_raw):
        for word_id in row:
            X[i] += to_categorical(word_id, num_classes=V)
        X[i] /= len(row) # Average context

    y = to_categorical(y_raw, num_classes=V)

    # Model Design
    model = Sequential([
        # The first layer's weights are the embeddings
        Dense(D, input_shape=(V,), activation='linear', name="embedding_layer"),
        Dense(V, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=lr), loss='categorical_crossentropy')
    
    # Training
    history = model.fit(X, y, epochs=epochs, verbose=1)

    # Save Loss
    with open(f'loss_cbow_{lr}.txt', 'w') as f:
        for loss in history.history['loss']:
            f.write(f"{loss}\n")

    # Export Embeddings
    weights = model.get_layer("embedding_layer").get_weights()[0]
    pd.DataFrame(weights).to_csv('embeddings_cbow.csv', index=False)
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    args = parser.parse_args()
    train_cbow(args.epochs, args.lr)