from operator import index
import re
import pandas as pd

#데이터프레임에 columns(카테고리) 추가 + 값 설정

df = pd.read_csv('./model/data_lastest2.csv')
value_counts = df['category'].value_counts()
print(value_counts)
#df['category'] = 'bill'
#df.to_csv('./model/1.bill_last.csv', index=False)
#
# def remove(title):
#
#     return re.compile(r'[_.\[\]]').sub(' ', title)
#
# df[df.columns[0]] = df[df.columns[0]].astype(str).apply(remove)
#
# # df = pd.read_csv('./model/naver_final.csv')
# df.to_csv('./model/3.promotion_ad_last_last.csv', index=False)


