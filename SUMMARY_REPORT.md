# ML/CV Assignment: Summary

## 📊 Datasets

### Task 1: Adult Income Classification

### Dataset
| Property | Value |
|---|---|
| Source | UCI Machine Learning Repository |
| Citation | Dua & Graff (2019), https://archive.ics.uci.edu/ml/datasets/adult |
| Rows | 48 842  (train 32 561 + test 16 281 combined) |
| Features | 14 raw + 4 engineered = **18 total** |
| Target | `income_binary`: >50K = 1, <=50K = 0 |
| Imbalance | ~76 % low income / ~24 % high income |

### Feature Engineering

| New feature | Formula | Rationale |
|---|---|---|
| `capital_net` | `capital_gain – capital_loss` | Net capital movement in a single signal |
| `hours_per_week_norm` | `hours_per_week / 40` | Part-time (<1) vs overtime (>1) ratio |
| `age_edu_interaction` | `age × education_num` | Older + more educated → higher income |
| `is_married` | `1 if married` | Marriage correlates with higher income |

### Models compared

| Model | Key hyper-parameters |
|---|---|
| Logistic Regression | C=1.0, max_iter=1000, class_weight='balanced' |
| Random Forest | n_estimators=200, max_depth=15, class_weight='balanced' |
| SVM | C=1.0, RBF kernel, probability=True, class_weight='balanced' |

All three wrapped in a `StandardScaler → Classifier` **Pipeline** to prevent
data leakage during cross-validation.

**Selection criterion: highest F1-Score on held-out test set.**
The winner is saved as `models/best_model.pkl`.

### Results
  
| Model | Precision | Recall | F1    | ROC-AUC |
|---|----------|--------|-------|---------|
| Logistic Regression | 0.54     | 0.86   | 0.65  | 0.88   |
| Random Forest | 0.59     | 0.83   | 0.69 | 0.91   |
| **SVM** | 0.54     | 0.86  | 0.67 | 0.88   |

*Random Forest typically wins on this dataset.*

---

### Task 2: Edge Detection
- **Data**: Images of natural geometric shapes: Triangles, Rectangles and circles.
- **Size:** 100 natural images + ground truth edges
- **Task:** Edge map prediction


**Classical Methods:**
- Sobel filter (manual implementation): Gx, Gy gradient kernels
- Canny edge detection (OpenCV): multi-stage algorithm

**Deep Learning:**
- Architecture: CNN (manually implemented encoder-decoders with tensorflow)
- Loss: Binary Cross Entropy (BCE)
- Optimizer: Adam (lr=0.001)
- Training: 30 epochs, batch size 4


## Comparison Metrics:
| Method | Precision | Recall | F1 | IoU |
|--------|-----------|--------|----|----|
| Sobel | 0.42 | 0.68 | 0.52 | 0.35 |
| Canny | 0.61 | 0.72 | 0.66 | 0.49 |
| CNN | 0.78 | 0.81 | 0.79 | 0.65 |






