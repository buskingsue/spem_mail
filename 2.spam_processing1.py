import pandas as pd

#초기 데이터
df = pd.read_csv('./data/output_file.csv', header=None, names=['spam'])

df1 = df[df[df.columns[0]].str.contains('광고')] # 광고 추출

#인덱스 삭제하고 저장
#광고라고 들어간 것 세이브
df.to_csv('./spam/filtered1.csv', index=False)
df1.to_csv('./spam/filtered_av.csv', index=False)
#====================================================================


