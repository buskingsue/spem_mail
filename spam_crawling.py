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


#========================================================================================
# CSV 파일 읽기
# utf-8 오류시 인코딩 시도 # ('경로', encoding=utf-8)

df = pd.read_csv('data/spam_mail.csv')
#df.drop_duplicates(inplace=True)
#df.reset_index(drop=True, inplace=True)

# 중복 제거
#df_no_duplicates = df.drop_duplicates()  # 중복된 행을 제거한 새로운 데이터프레임(df_no_duplicates)을 생성합니다.

# 중복 제거된 파일 저장
# df_no_duplicates.to_csv('spam_mail.csv', index=True)
# 중복을 제거한 데이터를 'spam_mail.csv'로 저장하며, 행 번호(index)는 포함하지 않습니다.



#========================================================================================

