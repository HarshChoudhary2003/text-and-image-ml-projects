# 🧠 Text & Image Machine Learning Projects  

This repository contains multiple **Machine Learning (ML)** and **Deep Learning (DL)** projects covering **Text Classification (NLP)** and some basic **Image ML experiments**.  
The highlight is **Spam Email Detection** using both **classical ML algorithms** and **modern Deep Learning models (LSTM with Keras/TensorFlow)**.  

---



## 📖 Introduction  

Spam emails are one of the biggest problems in digital communication.  
This project explores **different approaches to spam detection**:  

- **Classical Machine Learning:** Naive Bayes, Logistic Regression, MLPClassifier  
- **Deep Learning:** Embedding layers, LSTM neural networks with TensorFlow/Keras  
- **NLP Preprocessing:** Stopword removal, punctuation removal, tokenization, padding  
- **Visualization:** WordClouds, confusion matrix, accuracy/loss plots  

In addition, there are small **image-related ML experiments** using OpenCV (`cv2`).  

---

## ✨ Features  

- 📂 Load multiple datasets (`Emails.csv`, `spam.csv`, `synthetic_text_data.csv`)  
- 🧹 Clean and preprocess text data  
- ☁️ Generate **WordClouds** to visualize frequent words in spam vs ham emails  
- 🔡 Tokenization & sequence padding for deep learning models  
- 🧠 Train **classical ML models** and **deep learning LSTMs**  
- 📊 Evaluate with accuracy, precision, recall, F1-score, and confusion matrix  
- 📉 Plot training/validation accuracy & loss curves  
- 🖼️ Optional **image ML experiments** (OpenCV)  

---

## 📂 Project Workflow  

1. **Data Loading**  
   - Load datasets (`Emails.csv`, `spam.csv`, etc.)  

2. **Data Preprocessing**  
   - Lowercasing  
   - Removing punctuation & stopwords (NLTK)  
   - Tokenization & padding  

3. **Exploratory Data Analysis (EDA)**  
   - Class balance checks  
   - WordClouds for spam vs ham words  

4. **Modeling Approaches**  
   - **Baseline ML:** Naive Bayes, Logistic Regression, MLPClassifier  
   - **Deep Learning:** LSTM model with Embedding layers  

5. **Model Training**  
   - Train/test split  
   - EarlyStopping & ReduceLROnPlateau to prevent overfitting  

6. **Evaluation**  
   - Accuracy, precision, recall, F1-score  
   - Confusion matrix  
   - Accuracy/Loss plots  

7. **Image ML (Optional)**  
   - Small experiments with OpenCV  

---

## 📊 Datasets  

- **Emails.csv** → Main dataset for spam detection  
- **spam.csv** → Another labeled spam dataset (ham/spam)  
- **synthetic_text_data.csv** → Custom generated dataset for testing  

> 📌 Make sure you have these CSV files in your repo before running the notebook.  

---

## 🤖 Models Implemented  

### 🔹 Classical ML Models  
- Multinomial Naive Bayes  
- Logistic Regression  
- MLPClassifier (Sklearn Neural Network)  

### 🔹 Deep Learning Models (Keras/TensorFlow)  
- Embedding Layer + LSTM + Dense (Binary classification)  
- TextVectorization + Embedding + GlobalAveragePooling  
- Custom sequential & functional models  

---

## 📊 Results  

- ✅ Naive Bayes: Simple, fast baseline model  
- ✅ MLPClassifier: Performs better than baseline  
- ✅ LSTM (Deep Learning): Achieved **~90% accuracy** on test data  
- 📈 Accuracy/Loss plots confirm proper learning  
- ☁️ WordClouds show common spam & ham terms clearly  

---

## 🛠 Tech Stack  

- **Programming Language:** Python  
- **Data Handling:** Pandas, Numpy  
- **Visualization:** Matplotlib, Seaborn, WordCloud  
- **NLP:** NLTK, Scikit-learn (CountVectorizer, TfidfVectorizer)  
- **ML Models:** Naive Bayes, Logistic Regression, MLPClassifier  
- **Deep Learning:** TensorFlow / Keras (LSTM, Embedding, Dense layers)  
- **Image ML (Optional):** OpenCV (`cv2`)  

---

## ⚙️ Installation  

1. Clone the repo:  
   ```bash
   
   git clone(https://github.com/HarshChoudhary2003/text-and-image-ml-projects/tree/main)
   git clone
   cd text-image-ml-projects
