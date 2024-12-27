import pickle
import pandas as pd
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import numpy as np

#===================================================================
# 저장한 LabelEncoder 로드
with open('./model/encoder.pickle3', 'rb') as f:
    encoder = pickle.load(f)

label = encoder.classes_
print("Available Categories:", label)

# 모델 로드
model = load_model('./model/mail_category_model_test1_0.9770641922950745.h5')

# 스탑워드 로드
stopwords = pd.read_csv('./stopwords/stopwords.csv', index_col=0)

# 토크나이저 로드
with open('./model/mail_token_max_222.pickle', 'rb') as f:
    token = pickle.load(f)

# 형태소 분석기
okt = Okt()

#===================================================================
# 입력 및 예측 처리
while True:
    # 제목 입력
    title = input("메일 제목을 입력하세요 (종료하려면 STOP 입력): ")
    if title.upper() == "STOP":
        print("프로그램을 종료합니다.")
        break



    #===================================================================
    # 입력된 제목 전처리
    title_morphed = okt.morphs(title, stem=True)
    title_filtered = ' '.join([word for word in title_morphed if len(word) > 1 and word not in list(stopwords['stopword'])])
    title_tokenized = token.texts_to_sequences([title_filtered])
    title_padded = pad_sequences(title_tokenized, maxlen=222)

    #===================================================================
    # 모델 예측
    pred = model.predict(title_padded)
    predicted_category = label[np.argmax(pred)]
    prediction_probability = np.max(pred)  # 예측 확률
    accuracy = prediction_probability * 100  # 예측 확률을 퍼센트로 변환

    #===================================================================
    # 결과 출력
    print("\n[결과]")
    print(f"입력된 제목: {title}")
    print(f"예측된 카테고리: {predicted_category}")
    print(f"예측 확률: {prediction_probability:.4f} ({accuracy:.2f}%)\n")