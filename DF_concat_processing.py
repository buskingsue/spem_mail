import pandas as pd

# 스팸 합치기

df1 = pd.read_csv('./naver/filtered2.csv')
df2 = pd.read_csv('./spam/filtered_av.csv')
df3 = pd.read_csv('./naver/filtered2_2.csv')
df4 = pd.read_csv('./naver/filtered2_3.csv')
df5 = pd.read_csv('./spam/filtered1.csv')


df_1= pd.concat([df1,df2])
df_2= pd.concat([df_1,df3])
df_3= pd.concat([df_2,df4])
df_4= pd.concat([df_3,df5])

df_4.to_csv('./model/spam_final.csv', index=False)
#==========================================================

# 스팸X 합치기

df1 = pd.read_csv('./naver/filtered1.csv')
df2 = pd.read_csv('./naver/filtered1_2.csv')
df3 = pd.read_csv('./naver/filtered1_ds.csv')

df_concat = pd.concat([df1,df2,df3])

df_concat.to_csv('./model/naver_final.csv', index=False)

