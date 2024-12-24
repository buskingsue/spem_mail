import pandas as pd

df = pd.read_csv('data/spam_mail.csv')
#0번 콜론에 (광고)라는 단어가 들어간것 삭제
df_filtered = df[~df[df.columns[0]].str.contains(r'광고')]
df_filtered.info()

#광고라고 들어간 것 세이브
df_filtered.to_csv('filtered1.csv', index=False)
