# MovieLens Recommendation System with Transformer Models

This repository implements collaborative filtering recommendation models using transformer-based architectures (nanoGPT) and traditional embedding models (GMF, MLP, NeuMF) with TensorFlow 2.x and PyTorch. The models recommend movies based on user-item interactions from implicit feedback.

## Datasets
- **MovieLens 1M dataset (`ml-1m`)**: Contains 1 million user ratings.
- Located in the `Data/` directory.

## Models Implemented

### 1. Generalized Matrix Factorization (GMF)
- Linear embedding-based approach.
- Captures direct interactions through user-item embedding multiplication.

### 2. Multi-Layer Perceptron (MLP)
- Models non-linear user-item interactions using neural network layers.

### 3. Neural Matrix Factorization (NeuMF)
- Hybrid model combining GMF and MLP to capture both linear and non-linear user-item interactions.

### 4. nanoGPT Transformer
- Transformer-based model (GPT architecture) adapted for sequential movie recommendation.
- Treats movie sequences as tokenized input for prediction.

## Project Structure
```
.
├── Data/
│   └── ml-1m
├── GMF.py
├── MLP.py
├── NeuMF.py
├── movielens_nanogpt_format.txt
├── movielens_nanogpt.py
├── evaluate.py
├── Dataset.py
├── requirements.txt
└── README.md
```

## Installation
Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage
Run the desired model:

### GMF
```bash
python GMF.py
```

### MLP
```bash
python MLP.py
```

### NeuMF
```bash
python NeuMF.py
```

### Transformer (nanoGPT)
```bash
python movielens_nanogpt.py
```

## Data Preprocessing
The MovieLens dataset is processed into sequences compatible with transformer models (nanoGPT format), stored as tokenized movie sequences.

- Original dataset: `ml-1m/ratings.dat`
- Processed sequences saved to `movielens_nanogpt_format.txt`

## Configuration
Parameters (epochs, batch size, layers, learning rate, etc.) can be adjusted within each model script directly.

## Training & Evaluation
- Implements negative sampling for embedding models.
- Transformer model trained on sequences of movie IDs.
- Evaluates using **Hit Ratio (HR)** and **Normalized Discounted Cumulative Gain (NDCG)**.
- Model checkpoints saved in the respective output directories.

## Evaluation Metrics
- **Hit Ratio (HR)**: Checks if recommended items appear in the top-K recommended list.
- **Normalized Discounted Cumulative Gain (NDCG)**: Measures the ranking quality of recommendations.

Evaluation uses leave-one-out protocol for unbiased testing.

## Results
Performance metrics (HR, NDCG) are output during training. Models save weights of the best-performing epoch.

## Dependencies
- TensorFlow 2.x
- PyTorch
- NumPy
- SciPy
- Pandas

Install dependencies:
```bash
pip install -r requirements.txt
```

## Future Enhancements
- Hyperparameter tuning
- Advanced transformer architectures
- Docker container deployment

## References
- [Neural Collaborative Filtering, He et al. (2017)](https://dl.acm.org/doi/10.1145/3038912.3052569)

---

Contributions and suggestions are welcome! Submit pull requests or contact for inquiries.
