from unittest.mock import inplace
import pandas as pd
import numpy as np
import re

#

# 앞뒤 의미없는 글자 제거
#df = pd.read_csv('./data/A.csv', header=None, names=['mail'])

#df1 = df[df[df.columns[0]].str.contains('광고')] # 광고라는 단어가 포함된 columns[0]값 추출
#df = df[~df[df.columns[0]].str.contains('광고')] # 광고라는 단어가 포함된 columns[0]값 삭제


# df = df.replace(',', '')
# # 정규식을 사용해 여러 종류의 따옴표 제거

#df1.to_csv('./naver/filtered.csv', index=False)
# df_filtered.to_csv('./naver/filtered2-1.csv', index=False)

#===============================================================================
#
df = pd.read_csv('./data/naver_mail_2.csv', header=None, names=['mail'])
df = df[~df[df.columns[0]].str.contains('광고')]
df.to_csv('./naver/filtered1_2.csv', index=False)