import pandas as pd
import re


df_unclassified = pd.DataFrame([['떽떽','unclassified']],
                       columns=['mail','category'])

#df_promotion =
# df_social_business =
# df_info =
# df_unclassified =

df_unclassified.to_csv('./model/5.unclassified.csv', index=False)

# 콜론 드랍
# df = pd.read_csv('./model/clean_data_lv2.csv')
# df = df.drop(columns=['spam'])
# df.to_csv('./model/clean_data_lv3.csv', index=False)