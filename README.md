# Dota 2 Match Predictor

Predicts Radiant win/loss from draft hero compositions using logistic regression and a custom attention-based neural network.

## Why this problem is hard

Dota 2 has 130+ heroes with complex interdependencies. Hero synergy and counter-picking are non-linear — the value of any hero depends on the other nine picked. Draft-only prediction also ignores player skill, patch state, and in-game decision-making, so there's an inherent ceiling on accuracy from composition features alone.

## Pipeline

```
OpenDota API  →  MongoDB  →  Feature Extraction  →  Model Training
```

1. **Data collection** — pulls public match data from the OpenDota API and upserts into MongoDB
2. **Feature extraction** — each hero is embedded as a 37-dimensional vector combining:
   - One-hot: primary attribute (STR/AGI/INT), attack type (Melee/Ranged), 8 role flags
   - Normalized stats: base HP/mana/armor/damage, gain rates, attack/move speed, vision, pub win rate
   - Team flag (0 = Radiant, 1 = Dire)
   - Final input: 10 heroes × 37 dims = **370-dimensional match vector**
3. **Training** — two models trained and compared

## Models

### Logistic Regression
- `SGDClassifier` with mini-batch gradient descent, L2 regularization, optimal learning rate schedule
- Trained with `partial_fit` to handle large datasets without loading everything into memory

### Neural Network
- Per-team attention layers learn hero synergy within each draft
- Batch normalization + label smoothing (5%) to prevent overfit
- Adam optimizer with weight decay, early stopping on validation accuracy

## Results

| Model | Baseline (always predict Radiant) | Test Accuracy |
|---|---|---|
| Logistic Regression | ~54% | ~61% |
| Neural Network (attention) | ~54% | **~66%** |

The ~54% baseline reflects Radiant's slight structural win-rate advantage on public servers. Beating it by 12 points using only draft composition is meaningful given the information available.

## Feature engineering progression

| Attempt | Features | Accuracy |
|---|---|---|
| Raw hero IDs | scalar values | ~50% (random) |
| One-hot hero selection | sparse 2×N vector | ~60% |
| One-hot + role grouping | structured sparse | ~62% |
| Dense hero embeddings + attention NN | 370-dim dense | **~66%** |

## Saving and loading models

After training, save/load with:

```python
# Neural network
model.save("nn_model.pt")
loaded = NNModel.load("nn_model.pt", match_dim=74, lin_dim=64)

# Logistic regression
model.save("lr_model.pkl")
loaded = LGModel.load("lr_model.pkl")
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Requires a running MongoDB instance and an OpenDota API key set in a `.env` file:

```
MONGO_URI=mongodb://localhost:27017
```

Then run `dota2_models.ipynb` to train and evaluate both models.

## Stack

- **PyTorch** — neural network and attention mechanism
- **scikit-learn** — logistic regression, preprocessing, train/test split
- **MongoDB + pymongo** — match storage and retrieval
- **OpenDota API** — public match data source
- **Matplotlib** — training curve visualization
