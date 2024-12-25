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
#데이터 columns 추가 + 값 설정
#df = pd.read_csv('./model/naver_final.csv')
#df['spam'] = 'False'
#df1 = pd.read_csv('./model/spam_final.csv')
#df1['spam'] = 'True'

#df.to_csv('./model/clean_data.csv', index=False)
#df1.to_csv('./model/spam.csv', index=False)

df = pd.read_csv('./model/clean_data.csv')
#============================================================

#특수문자 제거

#특수문자 제거 함수 정의
def remove(title):
    return re.compile('[^가-힣A-Za-z ]').sub(' ', title)

#columns[0]에만 대입
df[df.columns[0]] = df[df.columns[0]].astype(str).apply(remove)

print(df.head())

df.to_csv('./model/clean_data_lv3.csv', index=False)

#============================================================

