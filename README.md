# 🎬 Human Action Classification from Motion Sequences

This project was completed as part of the *Learning of Structured Data* course portfolio, focusing on real-world human action recognition using motion sequence data.

## 📌 **Project Overview**
The dataset consists of motion sequences from 33 elderly participants performing five actions: **boxing, drums, guitar, rowing, and violin**.  
The goal was to build machine learning and deep learning models to classify these actions based on joint coordinate data.

## 🧰 **Pipeline Summary**
The project included:
- **Data loading & visualization**: Custom Python Dataset class to load and access motion sequences, plus visualization saved as MP4.
- **Data preprocessing**: Handling missing values and normalization (using StandardScaler).
- **Feature engineering**: Feature selection focusing on upper body joints, padding sequences for CNN input.
- **Model development**:
  - Traditional models: Random Forest, SVM, KNN, XGBoost.
  - Deep learning: 1D Convolutional Neural Network (CNN).
- **Hyperparameter tuning**: Using GridSearchCV and manual tuning.
- **Model evaluation**: Confusion matrices, accuracy, precision, recall, and F1-scores across multiple runs.
- **Deployment**: Best model saved with Joblib for reuse.

## 📊 **Results**
- CNN model achieved **mean accuracy of 83%** across 5 runs, outperforming traditional machine learning models.
- Feature selection on key joints improved both accuracy and training time.

## ⚙️ **Technical Stack**
- Python (Jupyter Notebook)
- Pandas, NumPy, Scikit-learn
- TensorFlow/Keras
- Matplotlib, Seaborn (for visualization)
- ffmpeg (for video creation)

## 📂 **Project Structure**
