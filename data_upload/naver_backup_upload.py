import os
import csv

# 폴더 경로 설정
folder_path = r"C:\Users\user\Downloads\data_test2\Files"

# 저장할 CSV 파일 경로
csv_file_path = r"../model/data_test2.csv"

# 폴더 안의 파일 이름을 추출
file_names = os.listdir(folder_path)

# CSV 파일로 저장
with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)

    # 헤더 추가 (선택 사항)
    writer.writerow(["mail"])

    # 파일 이름을 한 줄씩 CSV에 기록
    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):  # 파일인 경우만 저장
            writer.writerow([file_name])
