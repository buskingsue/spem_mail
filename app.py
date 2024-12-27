from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

# Flask 앱 생성
app = Flask(__name__)

# 사전 준비
model_path = "./model/spam_model_0.9863945841789246.h5"
tokenizer_path = "./model/mail_token_max_54.pickle"  # 모델 학습 시 사용된 토크나이저가 저장된 경로

# 모델 및 토크나이저 로드
model = tf.keras.models.load_model(model_path)
with open(tokenizer_path, "rb") as f:
    tokenizer = pickle.load(f)

# maxlen 값을 54로 설정 (학습 시 사용한 maxlen 값)
maxlen = 54

# 입력 텍스트를 모델에 전달 가능한 형식으로 변환
def preprocess_text(text):
    sequences = tokenizer.texts_to_sequences([text])  # 텍스트를 시퀀스로 변환
    padded = pad_sequences(sequences, maxlen=maxlen)  # 패딩 적용 (모델 학습 시 maxlen에 맞춤)
    return padded

# 라우트 설정
@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_result = None
    confidence = None
    title = None
    percent = None

    if request.method == 'POST':
        title = request.form['title']  # 폼 데이터 가져오기
        processed_title = preprocess_text(title)  # 입력 데이터 전처리
        prediction = model.predict(processed_title)  # 모델 예측
        # 예측 결과는 배열이므로 첫 번째 값을 사용하여 판단
        label = "Spam" if prediction[0][0] <= 0.1803 else "Not Spam"  # 첫 번째 예측 값 사용
        prediction_result = label
        confidence = round(100.0-(float(prediction[0][0]) * 100), 2)


    return render_template("index.html", prediction_result=prediction_result, confidence=confidence, title=title)

# Flask 앱 실행
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001, debug=False)  # 디버그 모드 활성화
