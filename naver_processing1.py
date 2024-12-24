import pandas as pd

df = pd.read_csv('data/naver_mail_ws.csv')

df[df.columns[0]] = df[df.columns[0]].str[:-3]
df[df.columns[0]] = df[df.columns[0]].str[13:]
print(df.head())

df.to_csv('filtered2.csv', index=False)