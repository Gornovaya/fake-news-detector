import os 
import pandas as pd 
import csv 
import numpy as np 
import nltk 
from nltk.stem import SnowballStemmer 
from nltk.stem.porter import PorterStemmer 
from nltk.tokenize import word_tokenize 
import seaborn as sb #для построение графиков 
import seaborn as sns 
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfTransformer 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.pipeline import Pipeline 
import nltk.corpus 
from gensim.models.word2vec import Word2Vec 
import matplotlib.pyplot as plt 
from collections import Counter 
import string # убрать пунктуацию 
from nltk.corpus import stopwords 
from sklearn.model_selection import train_test_split #для разделки датасета 
from google.colab import drive 

drive.mount('/content/drive') 
test_filename = '/content/drive/MyDrive/diplom/test.csv' 
train_filename = '/content/drive/MyDrive/diplom/train_valid.csv' 
train_news = pd.read_csv(train_filename, on_bad_lines='skip') 
test_news = pd.read_csv(test_filename, on_bad_lines='skip') 

nltk.download('stopwords') 
eng_stemmer = SnowballStemmer('english') 
stopwords = set(nltk.corpus.stopwords.words('english')) 
nltk.download('punkt') 
nltk.download('averaged_perceptron_tagger') 

# Дополнительная обработка данных 
def process_data(data, exclude_stopword=True, stem=True): 
  tokens = [w.lower() for w in data] 
  if stem: 
  tokens = [PorterStemmer().stem(token) for token in tokens] 
  if exclude_stopword: 
         stop_words = set(nltk.corpus.stopwords.words('english')) 
        tokens = [w for w in tokens if w not in stop_words] 
    # Remove punctuation 
    #tokens = [token for token in tokens if token not in string.punctuation] 
    return tokens 

# # Обработанные данные 
processed_train_data = process_data(train_news['Statement']) 
processed_test_data = process_data(test_news['Statement']) 

#logistic regression parameters 
parameters = {'LogR_tfidf__ngram_range': [(1, 1), (1, 2),(1,3),(1,4),(1,5)], 
               'LogR_tfidf__use_idf': (True, False), 
               'LogR_tfidf__smooth_idf': (True, False) 
} 
gs_clf = GridSearchCV(logR_pipeline_ngram, parameters, n_jobs=-1) 
gs_clf = gs_clf.fit(train_news['Statement'][:10000],train_news['Label'][:10000]) 

logR_pipeline_final = Pipeline([       
('LogR_tfidf',TfidfVectorizer(stop_words='english',ngram_range=(1,5),use_idf=Tru
e,smooth_idf=False)), 
        ('LogR_clf',LogisticRegression(penalty="l2",C=1)) 
        ]) 
logR_pipeline_final.fit(train_news['Statement'],train_news['Label']) 
predicted_LogR_final = logR_pipeline_final.predict(test_news['Statement']) 

# Вычислите точность модели 
accuracy = np.mean(predicted_LogR_final == test_news['Label']) 
print("Точность модели: ", accuracy) 

# Выведите отчет о классификации 
print(classification_report(test_news['Label'], predicted_LogR_final)) 
 
# Способы векторизации данных. 
nltk.download('treebank') 
#мешок слов 
countV = CountVectorizer(token_pattern=r"(?u)\b\w+\b") 
train_count = countV.fit_transform(processed_train_data) 
#tf-idf 
tfidfV = TfidfTransformer() 
train_tfidf = tfidfV.fit_transform(train_count) 
#Создание объекта TfidfVectorizer 
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7) 
#совмещенный метод 
countV_ngram = CountVectorizer(ngram_range=(1,3),stop_words='english') 
tfidf_ngram  = TfidfTransformer(use_idf=True,smooth_idf=True) 
tfidf_ngram = TfidfVectorizer(stop_words='english',ngram_range=(1,4),use_idf=True,smooth_idf=True) 
 
# Пример выбора классификатора на одном из методе векторизации CountV. 
nb_pipeline = Pipeline([ 
        ('NBCV', countV), 
        ('nb_clf',MultinomialNB())]) 
nb_pipeline.fit(train_news['Statement'], train_news['Label']) 
predicted_nb = nb_pipeline.predict(test_news['Statement']) 
np.mean(predicted_nb == test_news['Label']) 
 
logR_pipeline = Pipeline([ 
        ('LogRCV', countV), 
        ('LogR_clf',LogisticRegression()) 
        ]) 
logR_pipeline.fit(train_news['Statement'],train_news['Label']) 
predicted_LogR = logR_pipeline.predict(test_news['Statement']) 
np.mean(predicted_LogR == test_news['Label']) 
 
svm_pipeline = Pipeline([ 
        ('svmCV', countV), 
        ('svm_clf',svm.LinearSVC()) 
        ]) 
svm_pipeline.fit(train_news['Statement'], train_news['Label']) 
predicted_svm = svm_pipeline.predict(test_news['Statement']) 
np.mean(predicted_svm == test_news['Label']) 
 
random_forest = Pipeline([ 
        ('rfCV',countV), 
        ('rf_clf',RandomForestClassifier(n_estimators=200,n_jobs=3)) 
        ]) 
random_forest.fit(train_news['Statement'],train_news['Label']) 
predicted_rf = random_forest.predict(test_news['Statement']) 
np.mean(predicted_rf == test_news['Label']) 
 
#Функция для построения кривых обучения 
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)): 
    plt.figure() 
    plt.title(title) 
    if ylim is not None: 
     plt.ylim(*ylim) 
    plt.xlabel("Training examples") 
    plt.ylabel("Score") 
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes) 
    train_scores_mean = np.mean(train_scores, axis=1) 
    train_scores_std = np.std(train_scores, axis=1) 
    test_scores_mean = np.mean(test_scores, axis=1) 
    test_scores_std = np.std(test_scores, axis=1) 
    plt.grid() 
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, 
                     train_scores_mean + train_scores_std, alpha=0.1, 
                     color="r") 
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, 
                     test_scores_mean + test_scores_std, alpha=0.1, color="g") 
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", 
             label="Training score") 
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", 
             label="Cross-validation score") 
    plt.legend(loc="best") 
    return plt 
 
# Список моделей 
models = [nb_pipeline, logR_pipeline, svm_pipeline, random_forest] 
# Список названий моделей 
model_names = ['Naive Bayes', 'Logistic Regression', 'Linear SVM', 'Random Forest'] 
# Построение кривых обучения 
plot_learning_curve(models, model_names, train_news['Statement'], 
train_news['Label'], cv=5) 
plt.show() 
 
#Функция для вычисления оценки качества модели с использованием Stratified k-Fold 
def stratified_kfold_evaluation(pipeline, X, y, n_splits=5): 
    skf = StratifiedKFold(n_splits=n_splits) 
    scores = [] 
    for train_index, test_index in skf.split(X, y): 
        X_train, X_test = X[train_index], X[test_index] 
        y_train, y_test = y[train_index], y[test_index] 
        pipeline.fit(X_train, y_train) 
        predictions = pipeline.predict(X_test)
        score = f1_score(y_test, predictions, average='weighted')  # Указываем weighted для многоклассовой классификации 
        scores.append(score) 
    return np.mean(scores) 
 
#Метод подбора гиперпараметров RandomSearch. 
# Определение параметров для Random Search 
parameters = {'LogR_tfidf__ngram_range': [(1, 1), (1, 2),(1,3),(1,4),(1,5)], 
              'LogR_tfidf__use_idf': (True, False), 
              'LogR_tfidf__smooth_idf': (True, False) 
} 

# Инициализация RandomizedSearchCV 
random_search = RandomizedSearchCV(logR_pipeline_ngram, 
param_distributions=parameters, n_iter=100, cv=5, verbose=1, random_state=42, n_jobs=-1) 

# Обучение модель с помощью RandomizedSearchCV 
random_search.fit(train_news['Statement'], train_news['Label']) 

# Получение лучшие параметры 
best_params = random_search.best_params_ 
print("Лучшие параметры: ", best_params) 
logR_pipeline_final = Pipeline([       
('LogR_tfidf',TfidfVectorizer(stop_words='english',ngram_range=(1,5),use_idf=True,smooth_idf=False)), 
        ('LogR_clf',LogisticRegression(penalty="l2",C=1)) 
        ]) 

# Обучение модели с лучшими параметрами 
logR_pipeline_final.set_params(**best_params) 
logR_pipeline_final.fit(train_news['Statement'], train_news['Label']) 
 
#Функция выделки признаков классификатора. 
def show__features(model, vect, clf, text=None, n=50): 
    # Извлечкемк векторизатора и классификатора из конвейера 
    vectorizer = model.named_steps[vect] 
    classifier = model.named_steps[clf] 
    if not hasattr(classifier, 'coef_'): 
        raise TypeError( 
            "Cannot compute most informative features on {}.".format( 
                classifier.__class__.__name__ 
            ) 
        ) 
 
    if text is not None:
              # Вычисление коэффициенты для текста 
        tvec = model.transform([text]).toarray() 
    else: 
        # просто использовать коэффициенты 
        tvec = classifier.coef_ 
    # Архив названий объектов с коэффициентами и отсортировка 
    coefs = sorted( 
        zip(tvec[0], vectorizer.get_feature_names_out()), 
        reverse=True 
    ) 
    # Получаем первый n-й и послени1 n-й коэффициенты, называем пары 
    topn  = zip(coefs[:n], coefs[:-(n+1):-1]) 
    output = [] 
    # Если текст, добавляем прогнозируемое значение в выходные данные 
    if text is not None: 
        output.append("\"{}\"".format(text)) 
        output.append( 
            "Classified as: {}".format(model.predict([text])) 
        ) 
        output.append("") 
    # две колонки с наиболее негативными и наиболее позитивными признаками 
    for (cp, fnp), (cn, fnn) in topn: 
        output.append( 
            "{:0.4f}{: >15}    {:0.4f}{: >15}".format( 
                cp, fnp, cn, fnn 
            ) 
        ) 
    print(output)
