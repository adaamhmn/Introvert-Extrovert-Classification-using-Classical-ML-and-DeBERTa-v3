# Introvert-Extrovert-Classification-using-Classical-ML-and-DeBERTa-v3

This project builds a machine learning system that can predict **Introvert** or **Extrovert** personality traits from comments.  
The project compares how **traditional NLP + ML models** (TF-IDF, FastText, Logistic Regression, Random Forest) and a fine-tuned **DeBERTa-v3 transformer model** perform.

- Original dataset: [(MBTI) Myers-Briggs Personality Type Dataset](https://www.kaggle.com/datasets/datasnaek/mbti-type).
- Cleaned dataset can be downloaded [HERE](https://www.kaggle.com/datasets/adammuhaimin/mbti-preprocessed).

Main goal of this project is to evaluate how different modeling approaches handle:
- noisy Reddit-style text,
- personality language patterns,
- extreme class imbalance,
- and real-world writing style diversity.

---

## Project Features

### Classical Machine Learning Pipeline
- Text preprocessing  
- TF-IDF word & character n-grams  
- FastText embeddings  
- Dimensionality reduction using Truncated SVD
- Feature fusion (TF-IDF + FastText + length features)  
- Logistic Regression and Random Forest classifiers  
- Oversampling using **RandomOverSampler**  
- Macro-F1, Precision, Recall, Confusion Matrix evaluation
- Real-time inference function (`predict_personality`)  

### Transformer-Based Pipeline (DeBERTa-v3-base)
- Minimal text preprocessing  
- Tokenization using HuggingFace AutoTokenizer  
- Model: **microsoft/deberta-v3-base**  
- Imbalance handling using **WeightedRandomSampler**  
- Fine-tuning with HuggingFace Trainer  
- Macro-F1, Precision, Recall, Confusion Matrix evaluation
- Real-time inference function (`predict_personality`)

---

## About Dataset

- Cleaned version for Classical ML:
  - Replace '|||' separator with space
  - Lowercasing
  - Removing URLs
  - Removing MBTI type names to prevent data leakage
  - Removing Punctuation & Numbers
  - Converting Emoji to text
  - Removing stopwords
  -Lemmatizing 
  - Normalizing whitespace  
  
- Cleaned version for Transformer model:
  - Replace '|||' separator with space
  - Remove URLs
  - Removing MBTI type names to prevent data leakage
  - Normalizing whitespace  
  
- Final dataset labels:
  - **Introvert** → 6676 samples  
  - **Extrovert** → 1999 samples  
- Stored in:  
  `mbti_traditional_new.csv` (for classical ML)
  `mbti_deeplearning_new.csv` (for transformers)  

---

## Methods Overview

### **Classical NLP + ML Approach**
- TF-IDF word-level n-grams (1–2)
- TF-IDF char-level n-grams (3–5)
- FastText word embeddings
- TruncatedSVD (300 dims)
- Length-based linguistic features
- Models:
  - Logistic Regression (`class_weight="balanced"`)
  - Random Forest (`class_weight="balanced", n_estimators=300`)

---

### **Transformer Fine-Tuning (DeBERTa-v3-base)** 

Training configuration:
- `learning_rate = 2e-5`  
- `batch_size = 8–16`  
- `num_train_epochs = 3`  
- `fp16 = True`  
- `evaluation_strategy = "epoch"`  
- WeightedRandomSampler for balanced batches  

---

## Results

### **Logistic Regression**

| Metric           | Score    |
|------------------|----------|
| Accuracy         | **75%** |
| Macro F1         | **68%**  |
| Extrovert Recall | **64%**  |
| Introvert Recall | **78%**  |


### **Random Forest**
| Metric           | Score    |
|------------------|----------|
| Accuracy         | **77%** |
| Macro F1         | **54%**  |
| Extrovert Recall | **12%**  |
| Introvert Recall | **97%**  |


### **DeBERTa-v3 Performance**
| Metric           | Score     |
|------------------|-----------|
| Accuracy         | **85%**   |
| Macro F1         | **79.06%** |
| Extrovert Recall | **66%**   |
| Introvert Recall | **91%**   |

The transformer model significantly outperforms traditional ML models, especially for the minority class (**Extrovert**).

---

## Example Predictions

'Lmao that is hilarious! I literally shouted at my screen. We should totally do a meetup for this sub, it would be chaotic but fun.'
→ Extrovert

'Ugh, honestly I just want to stay in my room and play video games all weekend. People are so exhausting lol. Does anyone else feel like hiding when the doorbell rings'
→ Introvert

