import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model



df = pd.read_csv('./model/data_test.csv')
# df.drop_duplicates(inplace=True)
# df.reset_index(drop=True, inplace=True)

print(df.head())
df.info()
print(df.category.value_counts())

X = df['mail']
Y = df['category']
#===================================================================
# 저장한 더미화로 예측할 데이터 더미화
with open('./model/encoder.pickle', 'rb') as f:
    encoder = pickle.load(f)

label = encoder.classes_
print(label)

labeled_y = encoder.transform(Y)
onehot_Y = to_categorical(labeled_y)
print(onehot_Y)

#===================================================================
# 예측할 데이터 형태소 분리
okt = Okt()

for i in range(len(X)):
    X[i] = okt.morphs(X[i], stem=True)
print(X)

#===================================================================
# 스탑워드 제거

stopwords = pd.read_csv('./stopwords/stopwords.csv', index_col=0)
print(stopwords)

for sentence in range(len(X)):
    words = []
    for word in range(len(X[sentence])):
        if len(X[sentence][word]) > 1:
            if X[sentence][word] not in list(stopwords['stopword']):
                words.append(X[sentence][word])
    X[sentence] = ' '.join(words)

print(X[:5])
#===================================================================
# 제목 라벨링
# 모델 학습 시킬 때 나온 것

with open('./model/mail_token_max_222.pickle', 'rb') as f:
    token = pickle.load(f)

tokened_X = token.texts_to_sequences(X)
# print(tokened_X[:5])

 # 많을경우 자르기
for i in range(len(tokened_X)):
    if len(tokened_X[i]) > 222:
         tokened_X[i] = tokened_X[i][:222]
X_pad = pad_sequences(tokened_X, 222)
# print(X_pad[:5])

#========================================================================
# 모델 업로드
model = load_model('./model/mail_category_model_test1_0.9770641922950745.h5')
preds = model.predict(X_pad)

predicts = []
for pred in preds:
    most = label[np.argmax(pred)]
    # pred[np.argmax(pred)] = 0
    # second = label[np.argmax(pred)]
    # predicts.append([most, second])
    predicts.append(most)

df['predict'] = predicts

print(df[['category', 'predict']].head(30))

score = model.evaluate(X_pad, onehot_Y)
print(score[1])

df['OX'] = 0
for i in range(len(df)):
    if df.loc[i, 'category'] in df.loc[i, 'predict']:
        df.loc[i, 'OX'] = 1
print(df.OX.mean())