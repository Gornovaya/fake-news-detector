# Листинг кода итоговой модели MLP.  
import pandas as pd 
import nltk 
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from sklearn.model_selection import train_test_split 
from keras.models import Sequential 
from keras.layers import Dense, Dropout 
from keras.utils import to_categorical 
from keras.callbacks import EarlyStopping 
from sklearn.preprocessing import LabelEncoder 
from sklearn.metrics import accuracy_score 
!pip install pyspellchecker 
from spellchecker import SpellChecker 
import matplotlib.pyplot as plt 
import numpy as np 
from keras.utils import plot_model 

# Загрузка данных 
from google.colab import drive 
drive.mount('/content/drive') 
data_filename = '/content/drive/MyDrive/diplom/my_train.csv' 
df = pd.read_csv(data_filename) 
nltk.download('wordnet') 
nltk.download('punkt') 
nltk.download('stopwords') 
nltk.download('averaged_perceptron_tagger') 
stop_words = set(nltk.corpus.stopwords.words('english')) 
lemmatizer = WordNetLemmatizer() 

#создание словарей 
!pip install vaderSentiment 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 
import pandas as pd 

# Создание анализатора 
analyzer = SentimentIntensityAnalyzer() 

# Создание словаря эмоциональных слов 
emotion_words = set() 
for word in analyzer.lexicon: 
if analyzer.lexicon[word] != 0:  # Проверка, что слово имеет эмоциональную окраску 
  emotion_words.add(word) 
  
# Загрузка корпуса текстов 
nltk.download('webtext') 

# Извлечение текста из корпуса 
webtext_data = webtext.raw() 

# Создание словаря сленговых слов 
slang_words = set() 
for fileid in webtext.fileids(): 
      for word in webtext.words(fileid): 
        if word.isalpha() and word.islower():  # Пример фильтрации для сленговых слов 
            slang_words.add(word) 
 
# собственные словари 
emotion_words = ['Terrifying', 'Rapid', 'Struggling for Survival', 'Intense', 'Sustainable','Ambitious', … , 'Claim', 'Requires'] 
slang_words = ['fam', 'ballers', … ,'Spooke'] 
group_words = ['ALL', 'All', … , 'Everybody'] 
 
from nltk.stem import WordNetLemmatizer 
nltk.download('wordnet') 
nltk.download('punkt') 
from nltk.util import ngrams 
from nltk.tokenize import word_tokenize 

#обработка данных 
 def process_data(data, exclude_stopword=True, stem=True): 
     tokens = [w for w in data] #оставляем капс 
     if stem: 
         lemmatizer = WordNetLemmatizer() 
         tokens = [lemmatizer.lemmatize(token) for token in tokens] 
     if exclude_stopword: 
         stop_words = set(nltk.corpus.stopwords.words('english')) 
         tokens = [w for w in tokens if w.lower() not in stop_words] 
     return tokens 
processed_train_data = [process_data(text) for text in df['Statement']] 
 
# Выборка признаков 
def feature_selection(text): 
    tokens = process_data(text) 
    features = {} 
    features['exclamation_marks'] = text.count('!') > 2 
    features['text_in_caps'] = any(word.isupper() for word in tokens) 
    features['comparative_degree'] = any(word[1] == 'JJR' for word in nltk.pos_tag(tokens)) 
    features['superlative_degree'] = any(word[1] == 'JJS' for word in nltk.pos_tag(tokens)) 
    features['emotion_words'] = any(word in emotion_words for word in tokens) 
    features['slang_words'] = any(word in slang_words for word in tokens) 
    features['group_words'] = any(word in group_words for word in tokens) 
    spell = SpellChecker() 
    misspelled = spell.unknown(tokens) 
    features['spelling_errors'] = any(word in misspelled for word in tokens) 
    return features 

# Выборка признаков 
train_features = [feature_selection(text) for text in df['Statement']] 

# Признаки в DataFrame 
train_features_df = pd.DataFrame([list(feature.values()) for feature in train_features]) 

# Преобразование категориальных признаков в числовые 
train_features_df = train_features_df.apply(lambda x: x.astype(int) if x.dtype == 'bool' else x) 
train_features_df = train_features_df.apply(lambda x: x.astype(int) if x.dtype == 'float64' else x) 

# Векторизация меток 
df['Label'] = df['Label'].str.strip() 
le = LabelEncoder() 
y = le.fit_transform(df['Label']) 
np.save('/content/drive/MyDrive/diplom/le_classes.npy', le.classes_) 

# Раздел данных 
X_train, X_valid, y_train, y_valid = train_test_split(train_features_df, y, 
test_size=0.2, random_state=42) 

#Создание модели 
model = Sequential() 
model.add(Dense(32, input_dim=X_train.shape[1], activation='relu')) 
model.add(Dropout(0.2)) 
model.add(Dense(16, activation='relu')) 
model.add(Dropout(0.2)) 
model.add(Dense(1, activation='sigmoid')) 

# Структура модели 
plot_model(model, 'my_model.png', show_shapes=True) 
plot_model.show() 

#Компиляция 
model.compile(loss='binary_crossentropy', optimizer='adam',  metrics=['accuracy']) 

#Обучение 
early_stopping = EarlyStopping(monitor='val_loss', patience=3) 

history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_valid, y_valid), callbacks=[early_stopping]) 

# Предсказание 
valid_preds = model.predict(X_valid) 
valid_preds = (valid_preds > 0.5).astype(int) 
accuracy = accuracy_score(y_valid, valid_preds) 
print(f'Accuracy: {accuracy}') 

# Построение графиков кривых обучения 
plt.figure(figsize=(12, 6)) 
# График точности 
plt.subplot(1, 2, 1) 
plt.plot(history.history['accuracy']) 
plt.plot(history.history['val_accuracy']) 
plt.title('Model Accuracy') 
plt.ylabel('Accuracy') 
plt.xlabel('Epoch') 
plt.legend(['Train', 'Validation'], loc='upper left') 
# График потери 
plt.subplot(1, 2, 2) 
plt.plot(history.history['loss']) 
plt.plot(history.history['val_loss']) 
plt.title('Model Loss') 
plt.ylabel('Loss') 
plt.xlabel('Epoch') 
plt.legend(['Train', 'Validation'], loc='upper left') 
plt.tight_layout() 
plt.show() 

# Путь, где будет сохранена модель 
model_path = '/content/drive/MyDrive/diplom/my_model2.h5' 
# Сохранение модели 
model.save(model_path) 
