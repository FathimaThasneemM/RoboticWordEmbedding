import pandas as pd
import numpy as np
import argparse
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

def train_skipgram(epochs, lr):
    vocab = {}
    with open('vocab.txt', 'r') as f:
        for line in f:
            parts = line.split()
            vocab[parts[0]] = int(parts[1])
    V = len(vocab)
    D = 10

    df = pd.read_csv('skipgram_dataset.csv', header=None)
    X_raw = df[0].values
    y_raw = df[1].values

    X = to_categorical(X_raw, num_classes=V)
    y = to_categorical(y_raw, num_classes=V)

    model = Sequential([
        Dense(D, input_shape=(V,), activation='linear', name="embedding_layer"),
        Dense(V, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=lr), loss='categorical_crossentropy')
    history = model.fit(X, y, epochs=epochs, verbose=1)

    with open(f'loss_skipgram_{lr}.txt', 'w') as f:
        for loss in history.history['loss']:
            f.write(f"{loss}\n")

    weights = model.get_layer("embedding_layer").get_weights()[0]
    pd.DataFrame(weights).to_csv('embeddings_skipgram.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    args = parser.parse_args()
    train_skipgram(args.epochs, args.lr)