import pandas as pd
import numpy as np
from matplotlib.pyplot import title
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import pickle

from konlpy.tag import Okt

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from bs4 import  BeautifulSoup
import requests
import re
import pandas as pd
import datetime

#============================================================
df = pd.read_csv('model/data_lastest.csv')
X = df['mail']
Y = df['category']

# 열 전처리
# 카테고리(spam)더미화

encoder = LabelEncoder()
# fit_transform은 처음 한번만 해야한다.
labeled_y = encoder.fit_transform(Y)

label = encoder.classes_
print(label)

#Y = pd.get_dummies(Y)
#print(Y.head())

# 더미화 할 때 인코더의 라벨 정보를 파일로 저장.
with open('./model/encoder.pickle3', 'wb') as f:
     pickle.dump(encoder, f)

onehot_Y = to_categorical(labeled_y)
print(onehot_Y)

#======================================================
# 제목의 형태소 분리
df.info()

print(X[0])
okt = Okt()

#이게 안됐음
#for i in range(len(X)):
#    X[i] = okt.morphs(X[i], stem=True)

X = X.apply(lambda x: okt.morphs(x, stem=True))

print(X)
#======================================================
#%% stopwords 제거

# stopwords 목록을 CSV에서 불러옴.
stopwords_dir = './stopwords/stopwords.csv'
stopwords = pd.read_csv(stopwords_dir, index_col=0)
print(stopwords)


# 뉴스 제목들에서 위의 목록에 포함된 것들을 제거.
# for sentence: 하나의 문장을 인덱싱
# for word: 하나의 문장의 한 형태소를 인덱싱
for sentence in range(len(X)):

    words = []

    for word in range(len(X[sentence])):
        if len(X[sentence][word]) > 1:
            if X[sentence][word] not in list(stopwords['stopword']):
                # 글자수가 1보다 크고, stopword 목록에 없으면 리스트에 추가.
                words.append(X[sentence][word])

    # 리스트의 형태소들을 공백하나로 분리하여 하나의 문장으로 합쳐서 다시 저장.
    X[sentence] = ' '.join(words)


print(X[:5])
#======================================================
#형태소 숫자로 라벨링
token = Tokenizer()

token.fit_on_texts(X)

tokened_X = token.texts_to_sequences(X)

# 단어 개수
# 패딩을 하기 위해서 +1로 토큰화된 개수를 늘려줌.
wordsize = len(token.word_index) + 1

print(tokened_X[:5])
#======================================================

# 제일 긴 문장을 찾고 나머지 문장을 0으로 채운다
max = 0
for i in range(len(tokened_X)):
    if max < len(tokened_X[i]):
        max = len(tokened_X[i])

print(max)

X_pad = pad_sequences(tokened_X, max)
print(X_pad)

with open(f'./model/mail_token_max_{max}.pickle', 'wb') as f:
    pickle.dump(token, f)
#=======================================================

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X_pad, onehot_Y, test_size=0.1)
print(Xtrain.shape, Ytrain.shape)
print(Xtest.shape, Ytest.shape)

np.save(f'./npdata/1mail_data_X_train_max_{max}_wordsize_{wordsize}', Xtrain)
np.save(f'./npdata/1mail_data_Y_train_max_{max}_wordsize_{wordsize}', Ytrain)
np.save(f'./npdata/1mail_data_X_test_max_{max}_wordsize_{wordsize}', Xtest)
np.save(f'./npdata/1mail_data_Y_test_max_{max}_wordsize_{wordsize}', Ytest)