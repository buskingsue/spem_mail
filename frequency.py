import pandas as pd
from collections import Counter
import re

# CSV 파일 읽기
df = pd.read_csv('model/backup/clean_data_ANNE.csv')

# 데이터프레임의 모든 텍스트를 하나로 합치기 (예: 첫 번째 컬럼만 분석)
text_data = ' '.join(df[df.columns[0]].astype(str))

# 텍스트 전처리 (특수 문자 제거 및 소문자로 변환)
processed_text = re.sub(r'[^가-힣a-zA-Z\s]', '', text_data).lower()

# 단어 단위로 분리
words = processed_text.split()

# 단어 빈도 계산
word_counts = Counter(words)

# 상위 100개 단어 추출
top_100_words = word_counts.most_common(100)

# 결과 출력
print("상위 100개 단어:")
for word, count in top_100_words:
    print(f"{word}: {count}")