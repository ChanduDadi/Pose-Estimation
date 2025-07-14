# Motion Sequence Action Classification

## 📌 Overview
This project is part of our Learning of Structured Data (LSD) coursework (Portfolio Part 3).  
It focuses on building a classifier to predict human actions (boxing, drums, guitar, rowing, violin) from motion sequence data of elderly people playing charades.  
The pipeline includes:
- Data loading and visualization
- Preprocessing
- Feature engineering
- Model design (traditional ML & deep learning)
- Evaluation and analysis

## 📂 Dataset
The dataset includes motion sequences in CSV format, split into:
- `data/train/` : 27 subjects (training data)
- `data/test/` : 6 subjects (test data)

Each file contains sequences of joint coordinates for a subject performing an action.

## ⚙️ Methodology

### Data Loading & Visualization
- Implemented a Python `Dataset` class:
  - `getSequence(dataset, subject, action, iteration)`  
  - `random(dataset)`
  - `visualize(dataset, subject, action, iteration, frame)` to generate frame-wise skeleton visualizations.
- Visualizations of sequences were converted to videos using FFmpeg.

### Preprocessing
- Handled missing/NaN values.
- Standardized features (zero mean, unit variance) for models sensitive to scale (e.g., SVM, KNN).
- Feature selection focused on joints with higher discriminative power (arms, wrists, shoulders, elbows).

### Feature Engineering
- Used padding (`pad_sequences`) to handle variable-length sequences.
- Included joint angles and positions for richer feature sets.

### Model Development
Explored:
- **Traditional ML**: Random Forest, SVM, KNN, XGBoost
- **Deep Learning**: 1D CNN with multiple convolutional and pooling layers

### Hyperparameter Tuning
- Used `GridSearchCV` for traditional models.
- Manually tuned CNN hyperparameters.

### Evaluation
- Metrics: Accuracy, confusion matrix, precision, recall, F1-score.
- Performed 5 independent runs to report mean ± standard deviation.

### Results
| Model | Accuracy (mean) | Std Dev |
|------|-----------------|--------:|
| CNN  |       0.83      |   0.02  |

The CNN consistently achieved >80% accuracy.

### Time & Memory Complexity
- Linear time and memory complexity with respect to sample size.

## 🛠️ Technical Stack
- Python
- Pandas, NumPy, Scikit-learn, TensorFlow
- Matplotlib, Seaborn
- FFmpeg
- Joblib (model serialization)

## 📊 Deliverables
- Code: Python scripts & Jupyter notebook
- Report: PDF explaining methodology and results
- Video: Sequence visualization

## 👥 Authors
- Anusha Salimath
- Chandu Dadi
- Priyanka Hareshbhai Sorathiya

## 📄 License
[Specify license, e.g., MIT]

## ✏️ How to run
1. Clone the repo:
    ```bash
    git clone https://github.com/username/repo-name.git
    cd repo-name
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the notebook / scripts to train models and visualize results.

---

> *This project was submitted as part of the Master's coursework at the Faculty of Computer Science and Business Informatics.*
