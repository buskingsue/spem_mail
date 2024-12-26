import pandas as pd
import numpy as np

from keras._tf_keras.keras.utils import to_categorical
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
from keras._tf_keras.keras.models import load_model

import pickle
from konlpy.tag import Okt

model = load_model('./model/spam_model_0.9863945841789246.h5')

with open('./model/mail_token_max_54.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

okt = Okt()


# 2. 문장 스팸 여부 예측 함수
def predict_spam(input_sentence):
    tokenized_sentence = okt.morphs(input_sentence, stem=True)
    sequence = tokenizer.texts_to_sequences([tokenized_sentence])
    max_length = 100
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')
    prediction = model.predict(padded_sequence)

    # prediction이 배열일 경우 첫 번째 값을 추출
    spam_probability = prediction[0][0]  # 2D 배열에서 첫 번째 값을 가져옴

    if spam_probability < 0.5:
        return "스팸 메시지입니다."
    else:
        return "정상 메시지입니다."


# 3. 테스트
test_sentence = "(광고) 역대급 강력한 AI 폴더블 폰의 탄생"
result = predict_spam(test_sentence)
print(f"입력 문장: {test_sentence}")
print(f"예측 결과: {result}")