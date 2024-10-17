#File này được sử dụng nhằm xóa bớt dữ liệu trong các tệp dữ liệu
import os
import random

def delete_random_files(folder_path, num_files_to_delete):
    files = os.listdir(folder_path)
    
    files_to_delete = random.sample(files, num_files_to_delete)
    
    for file in files_to_delete:
        file_path = os.path.join(folder_path, file)
        os.remove(file_path)
        print(f"Đã xóa: {file_path}")

delete_random_files('E:/Homework/Machine_Learning/Human_emotion_dectection/images/train/sad', 328)
