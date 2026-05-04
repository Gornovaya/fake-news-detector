# fake-news-detector
Fake news detection in headlines using Classificators and Neyron mpdels. Includes handcrafted features, custom dataset, and two approaches — context-based and feature-based classification.

# Fake News Detection (Headline Classification)

## Overview

This project was developed as part of a bachelor's thesis. It focuses on binary classification of news headlines into TRUE or FALSE categories. Two classification approaches are implemented and compared:

1. Context-based classification using Logistic Regression with TF-IDF vectorization and n-grams.
2. Feature-based classification using a Multi-Layer Perceptron (MLP) with handcrafted linguistic features.

## Key Components

- **Logistic Regression** – context-based approach using TF-IDF (n-grams 1–5) and stop word filtering.
- **MLP (Multi-Layer Perceptron)** – neural network trained on manually defined headline features.
- **LSTM (experimental)** – tested but outperformed by the MLP architecture.
- **Custom dataset** – 500 headlines (80/20 train/test split) collected from verified news sources (lenta.ru, government.ru) and fact-checked false news (2020–2024).
- **Public dataset** – LIAR dataset (12.8k statements) used for initial training and validation.
- **Feature engineering** – includes detection of exclamation marks, caps lock, comparative/superlative degrees, emotion words, slang, group generalizations, and spelling errors.

## Model Performance (Test Set)

| Model | Approach | Accuracy |
| :--- | :--- | :--- |
| Logistic Regression | Context (TF-IDF, n-grams 1–5) | 75% |
| MLP | Handcrafted features + Dense layers | 79% |
| LSTM | Features + Embedding + LSTM | ~69% |

Based on these results, the MLP model was selected as the final solution. The feature-based approach with manually defined linguistic indicators provided the highest classification accuracy.

## Dataset Description

Two datasets were used in this study:

1. **LIAR dataset** – a public benchmark for fake news detection, containing 12.8k labeled statements. Used for context-based classification.
2. **Custom dataset** – 500 headlines collected and labeled manually. The dataset includes real headlines from official sources (lenta.ru, government.ru) and false headlines based on fact-checked misinformation from 2020–2024. The dataset is balanced and split 80/20 for training and testing.

The feature set was designed specifically for the custom dataset and includes the following binary indicators:

- Caps lock presence
- Excessive exclamation marks (>2)
- Comparative and superlative degree words
- Emotionally charged words
- Slang expressions
- Group-generalizing words (e.g., "everybody", "all")
- Spelling errors (detected using pyspellchecker)

## Installation and Usage

Clone the repository and install dependencies:

```bash
git clone https://github.com/Gornovaya/fake-news-detector.git
cd fake-news-detector
pip install -r requirements.txt
```

Run the corresponding notebook or script depending on the desired approach:
```bash
logistic_regression.ipynb – context-based classification
mlp_model.ipynb – feature-based classification with MLP
```

Each script accepts a headline as input and returns a prediction (TRUE / FALSE) along with a confidence score.

## Dependencies:

- pandas
- numpy
- scikit-learn
- nltk
- tensorflow
- keras
- pyspellchecker
- vaderSentiment
- matplotlib
- seaborn

NLTK additional data is required:
```bash
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('webtext')
```

## Project Structure
fake-news-detector/

├── data/

│   ├── train.csv

│   ├── test.csv

│   └── my_train.csv

├── notebooks/

│   ├── logistic_regression.ipynb

│   ├── mlp_model.ipynb

│   └── lstm_model.ipynb

├── models/

│   ├── logistic_model.pkl

│   ├── mlp_model.h5

│   └── le_classes.npy

├── requirements.txt

├── README.md

└── LICENSE



