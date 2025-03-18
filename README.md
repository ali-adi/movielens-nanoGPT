# MovieLens nanoGPT Transformer Recommendation System

This repository implements a transformer-based recommendation system inspired by GPT architectures (nanoGPT) using PyTorch. The model recommends movies by treating sequences of user interactions as tokenized inputs, making predictions based on previously viewed movies.

## Dataset
- **MovieLens 1M dataset (`ml-1m`)**: Contains 1 million user ratings.
- Data processed into sequential format for nanoGPT.

## Model Implemented

### nanoGPT Transformer
- Adapted GPT architecture for sequential movie recommendation.
- Processes sequences of movies watched by users as tokens to predict subsequent movies.

## Project Structure
```
.
├── Data/
│   └── ml-1m
├── movielens_nanogpt_format.txt
├── movielens_nanogpt.py
├── requirements.txt
└── README.md
```

## Installation
Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage
To run the nanoGPT Transformer model:
```bash
python movielens_nanogpt.py
```

## Data Preprocessing
MovieLens dataset converted to sequential nanoGPT format:
- Sequences are sorted by user and timestamp.
- Saved to `movielens_nanogpt_format.txt`.

## Configuration
Adjust hyperparameters (batch size, learning rate, embedding size, etc.) directly within `movielens_nanogpt.py`.

## Training & Evaluation
- Model trained using movie sequences.
- Evaluation periodically reports training and validation losses.

## Dependencies
- PyTorch
- NumPy
- Pandas

Install dependencies:
```bash
pip install -r requirements.txt
```

## Future Enhancements
- Implement additional sequence-aware metrics.
- Hyperparameter tuning.
- Integration into production via containerization (Docker).

## References
- [nanoGPT repository](https://github.com/karpathy/nanoGPT)

---

Contributions and suggestions are welcome! Submit pull requests or contact for inquiries.
