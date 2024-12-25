
Y = df['category']

encoder = LabelEncoder()
# fit_transform은 처음 한번만 해야한다.
labeled_y = encoder.fit_transform(Y)
print(labeled_y[:3])

label = encoder.classes_
print(label)

# Y = pd.get_dummies(Y)
# print(Y.head())


# # 더미화 할 때 인코더의 라벨 정보를 파일로 저장.
# with open('../models/encoder.pickle', 'wb') as f:
#     pickle.dump(encoder, f)
#
# onehot_Y = to_categorical(labeled_y)
# print(onehot_Y)
