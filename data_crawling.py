#프로젝트 이름
#설명(Description)
#목차(Table of Contents)
#설치(Installation)
#용례(Usage)
#설정(Configuration)
#라이센스(License)
#참조(Credits/Acknowledgments)
import pandas as pd  # pandas 라이브러리를 불러옵니다.
import pickle
import pandas as pd


# CSV 파일 읽기
# utf-8 인코딩 시도
# utf-8 인코딩 시도
df = pd.read_csv('./data/spam_mail.txt', encoding='utf-8')
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)
print(df.head())
# 중복 제거
df_no_duplicates = df.drop_duplicates()  # 중복된 행을 제거한 새로운 데이터프레임(df_no_duplicates)을 생성합니다.
df.info()
# 중복 제거된 파일 저장
df_no_duplicates.to_csv('output_file.csv', index=False)
# 중복을 제거한 데이터를 'output_file.csv'로 저장하며, 행 번호(index)는 포함하지 않습니다.

print("중복 제거 완료. 결과는 'output_file.csv'에 저장되었습니다.")
# 작업이 완료되었음을 알립니다.
