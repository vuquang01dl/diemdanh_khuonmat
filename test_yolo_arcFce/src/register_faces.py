#  đăng ký khuôn mặt

import os
import sys
import numpy as np

# Thêm đường dẫn thư mục src để đảm bảo có thể import module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from arcface import extract_embedding
DATABASE_PATH = r"D:\nhan_dien_khuon_mat_chinh_xac_cao\test_yolo_arcFce\data\registered_faces"
EMBEDDINGS_PATH = r"D:\nhan_dien_khuon_mat_chinh_xac_cao\test_yolo_arcFce\data\faces-embeddings.npz"

def register_faces():
    """ Đăng ký các khuôn mặt có sẵn trong thư mục registered_faces """
    face_db = {}
    
    for filename in os.listdir(DATABASE_PATH):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            name = os.path.splitext(filename)[0]  # Lấy tên file
            embedding = extract_embedding(os.path.join(DATABASE_PATH, filename))
            if embedding is not None:
                face_db[name] = embedding

    np.savez(EMBEDDINGS_PATH, **face_db)  # Lưu embeddings vào file
    print(f"✅ Đã lưu {len(face_db)} khuôn mặt vào database.")

if __name__ == "__main__":
    register_faces()
