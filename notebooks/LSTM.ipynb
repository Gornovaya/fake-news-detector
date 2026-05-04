# Листинг кода итоговой модели LSTM.  
import pandas as pd 
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences 
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, RepeatVector, TimeDistributed 
from keras.layers import Embedding, LSTM, Dense 
from keras.utils import to_categorical 
import numpy as np 
from sklearn.model_selection import train_test_split 
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences 
from keras.models import Sequential 
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D 
from keras.utils import to_categorical 
import re 
from sklearn.preprocessing import LabelEncoder 
import matplotlib.pyplot as plt 
from keras.layers import Input, concatenate 
from keras.models import Model 
from keras.utils import plot_model 
from google.colab import drive 
drive.mount('/content/drive') 
data_filename='/content/drive/MyDrive/diplom/my_train.csv' 
data = pd.read_csv(data_filename) 
# Подготовка данных 
data['Statement'] = data['Statement'].apply(lambda x: x.lower()) 
data['Statement'] = data['Statement'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', 
'', x))) 
#Токенизация текста 
max_features = 2000 
tokenizer = Tokenizer(num_words=max_features, split=' ') 
tokenizer.fit_on_texts(data['Statement'].values) 
X = tokenizer.texts_to_sequences(data['Statement'].values) 
X = pad_sequences(X) 
embed_dim = 128 
lstm_out = 196 
# Векторизация меток 
labelencoder = LabelEncoder() 
y = labelencoder.fit_transform(data['Label']) 
y = to_categorical(y) 
#создание словарей 
!pip install vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 
import pandas as pd 
# Создание анализатора 
analyzer = SentimentIntensityAnalyzer() 
# Создание словаря эмоциональных слов 
emotion_words = set() 
for word in analyzer.lexicon: 
    if analyzer.lexicon[word] != 0:  # Проверка, что слово имеет эмоциональную 
окраску 
        emotion_words.add(word) 
# Загрузка корпуса текстов 
nltk.download('webtext') 
# Извлечение текста из корпуса 
webtext_data = webtext.raw() 
# Создание словаря сленговых слов 
slang_words = set() 
for fileid in webtext.fileids(): 
    for word in webtext.words(fileid): 
        if word.isalpha() and word.islower():  # Пример фильтрации для сленговых 
слов 
            slang_words.add(word) 
 
# собственные словари 
emotion_words = ['Terrifying', 'Rapid', 'Struggling for Survival', 'Intense', 
'Sustainable','Ambitious', … , 'Claim', 'Requires'] 
slang_words = ['fam', 'ballers', … ,'Spooke'] 
group_words = ['ALL', 'All', … , 'Everybody'] 
 
# Выборка признаков 
def feature_selection(text): 
    *аналогично MLP* 
    return features 
 
# Применение функции feature_selection к тексту 
data['features'] = data['Statement'].apply(feature_selection) 
# Преобразование признаков в числовые данные 
feature_data = pd.concat(list(data['features']), ignore_index=True) 
# Объединение признаков с данными о тексте 
X = np.concatenate((X, feature_data.values), axis=1) 
# Разделение данных на обучающие и тестовые наборы 
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, 
random_state=42)

# Разделение признаков на обучающие и тестовые наборы 
feature_train, feature_test = train_test_split(feature_data, test_size=0.2, 
random_state=42) 
# Создание новой модели, которая будет принимать дополнительные признаки 
input_features = Input(shape=(feature_train.shape[1],)) 
# Добавление слоев в модель 
x 
= 
Embedding(max_features, 
input_length=feature_train.shape[1])(input_features) 
x = SpatialDropout1D(0.4)(x) 
x = LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2)(x) 
output = Dense(2, activation='softmax')(x) 
# Создание модели 
model = Model(inputs=input_features, outputs=output) 
# Рисование структуры модели 
plot_model(model, 'my_model.png', show_shapes=True) 
plot_model.show() 
# Компиляция модели 
model.compile(loss='categorical_crossentropy', 
metrics=['accuracy']) 
# Обучение модели 
embed_dim, 
optimizer='adam', 
history = model.fit(feature_train, Y_train, epochs=15, batch_size=32, verbose=2) 
# Сохранение модели на диск 
model.save('/content/drive/MyDrive/diplom/my_model.h5') 
# Оценка модели 
score, acc = model.evaluate([X_test, feature_test], Y_test, verbose=2, 
batch_size=32) 
# Графики эффективности обучения модели 
plt.plot(history.history['accuracy']) 
plt.title('Model Accuracy') 
plt.ylabel('Accuracy') 
plt.xlabel('Epoch') 
plt.legend(['Train'], loc='upper left') 
plt.show() 
plt.plot(history.history['loss']) 
plt.title('Model Loss') 
plt.ylabel('Loss') 
plt.xlabel('Epoch') 
plt.legend(['Train'], loc='upper left') 
plt.show() 
