from insightface.app import FaceAnalysis
import numpy as np
import cv2

# Load model ArcFace (buffalo_l có sẵn trong insightface)
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0)  # Chạy trên CPU

def extract_embedding(image_path):
    """ Trích xuất embedding từ ảnh """
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Không thể đọc ảnh: {image_path}")
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    faces = app.get(img)
    if len(faces) == 0:
        print("⚠️ Không tìm thấy khuôn mặt!")
        return None

    print("✅ Tìm thấy khuôn mặt, trích xuất embedding...")
    return faces[0].normed_embedding
