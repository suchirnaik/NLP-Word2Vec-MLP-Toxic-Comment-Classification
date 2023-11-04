# Observations and Analysis

For this implementation, Twitter 25-word embeddings were used. The choice of these embeddings was made to prevent long computations and kernel failures due to increased layers in the model. The maximum comment length was set to 40 to reduce the computational load.

## Observations

### Twitter 25 Embeddings

| Sl.no | Model              | Accuracy | Precision | Recall | F1-score |
|-------|--------------------|----------|-----------|--------|----------|
| 1.    | 1-layer MLP model  | 88.94    | 41.3      | 38.4   | 39.8     |
| 2.    | 2-layer MLP model  | 88.69    | 41.07     | 43.26  | 42.14    |
| 3.    | 3-layer MLP model  | 89.11    | 42.51     | 40.70  | 41.59    |

## Analysis

- All three models perform similarly with slight variations in accuracy.
- The 3-layer MLP model achieves the highest accuracy, but other scores like precision, recall, and F1-score indicate the 2-layer MLP model performs best.
- The 2-layer MLP model has the highest recall (43.26%), and the 3-layer MLP model has the highest precision (42.51%).
- The overall best performer is the 2-layer MLP model with the highest F1-score.

## Individual Model Analysis

### 1-layer MLP model:

- Precision: 41.3%
- Recall: 38.4%
- F1-score: 39.8%

This model has the lowest performance among the three.

### 2-layer MLP model:

- Precision: 41.07%
- Recall: 43.26%
- F1-score: 42.14%

This model performs the best with the highest F1-score.

### 3-layer MLP model:

- Precision: 42.51%
- Recall: 40.70%
- F1-score: 41.59%

The 3-layer MLP model has the highest precision.

## Accuracy of Toxic and Non-toxic Labels Separately

- Toxic class accuracy: 38.44% (1-layer), 43.27% (2-layer), 40.71% (3-layer)
- Non-toxic class accuracy: 94.25% (1-layer), 93.47% (2-layer), 94.20% (3-layer)

## Additional Observations

- Each run of the MLP models resulted in different scores due to random weight initialization.
- Setting the `random_state` parameter in the MLP classifier can mitigate this randomness.
- Generally, models with more layers performed better in predicting the toxic class over multiple runs.

