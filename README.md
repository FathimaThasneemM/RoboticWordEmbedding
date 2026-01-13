Robotic Word Embedding Project (10-D)

This project learns 10-dimensional word embeddings from a robotic corpus using CBOW and Skip-gram architectures in Keras/TensorFlow.
Steps:
1. Installation
Ensure you have the following installed in your virtual environment:
- Python 3.12+
- TensorFlow 2.18+
- Pandas, Numpy, Scikit-learn

2. Execution Steps
Run the scripts in the following order:

1. Build Dataset:`python dataset_builder.py`
2. Train CBOW: `python train_cbow.py --lr 0.01 --epochs 100`
3. Train Skip-gram: `python train_skipgram.py --lr 0.01 --epochs 100`
4. Evaluate Results: `python evaluate_embeddings.py`

3. Parameter Settings
- Embedding Dimension (D): 10 (Fixed via hidden layer units)
- Context Window (W): 4
- Epochs (E): 100
- Learning Rates Tested:0.1, 0.01, 0.001

4. Project Files
- `vocab.txt`: Word IDs and frequencies.
- `cbow_dataset.csv` / `skipgram_dataset.csv`: Generated training pairs.
- `loss_*.txt`: Training loss logs for convergence analysis.
- `embeddings_*.csv`: The learned 10-D weight matrices.
- `similarity_results.txt`: Top-5 nearest neighbors for query words.