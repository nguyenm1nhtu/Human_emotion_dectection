import os
import random

def delete_random_files(folder_path, num_files_to_delete):
    # Lấy danh sách tất cả các file trong thư mục
    files = os.listdir(folder_path)
    
    # Chọn ngẫu nhiên số lượng file cần xóa
    files_to_delete = random.sample(files, num_files_to_delete)
    
    # Xóa các file đã chọn
    for file in files_to_delete:
        file_path = os.path.join(folder_path, file)
        os.remove(file_path)
        print(f"Đã xóa: {file_path}")

# Sử dụng hàm này
delete_random_files('E:/Homework/Machine_Learning/Human_emotion_dectection/images/train/sad', 328)  # Ví dụ xóa 10 file ngẫu nhiên trong thư mục
