import cv2
import os
import numpy as np
import onnxruntime
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
from tkinter import messagebox, simpledialog
from PIL import Image, ImageTk

# Đường dẫn lưu embeddings
EMBEDDINGS_PATH = r"D:\nhan_dien_khuon_mat_chinh_xac_cao\test_yolo_arcFce\data\faces-embeddings.npz"
SAVE_DIR = r"D:\\nhan_dien_khuon_mat_chinh_xac_cao\\test_yolo_arcFce\\data\\registered_faces"
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0)

def load_database():
    if not os.path.exists(EMBEDDINGS_PATH):
        return [], np.array([])
    data = np.load(EMBEDDINGS_PATH, allow_pickle=True)
    return list(data.keys()), np.array(list(data.values()))

def capture_images(username):
    os.makedirs(SAVE_DIR, exist_ok=True)
    cap = cv2.VideoCapture(0)
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Capture Image", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            file_path = os.path.join(SAVE_DIR, f"{username}_{count}.jpg")
            cv2.imwrite(file_path, frame)
            count += 1
        elif key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def extract_embedding(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = app.get(img)
    if not faces:
        return None, None
    return [face.normed_embedding for face in faces], [face.bbox for face in faces]

def recognize_face(frame, names, embeddings):
    face_embeddings, bboxes = extract_embedding(frame)
    if face_embeddings is None:
        return []
    results = []
    for emb, bbox in zip(face_embeddings, bboxes):
        similarities = cosine_similarity([emb], embeddings)[0]
        best_match_idx = np.argmax(similarities)
        best_score = similarities[best_match_idx]
        name = names[best_match_idx] if best_score > 0.6 else "Unknown"
        results.append((name, bbox))
    return results

def start_recognition():
    names, embeddings = load_database()
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        faces = recognize_face(frame, names, embeddings)
        for name, bbox in faces:
            x1, y1, x2, y2 = map(int, bbox)
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def login():
    username = simpledialog.askstring("Login", "Nhập tên của bạn:")
    if username:
        messagebox.showinfo("Login", f"Xin chào {username}! Bắt đầu chụp ảnh để train.")
        capture_images(username)

def main():
    root = tk.Tk()
    root.title("Face Recognition System")
    root.geometry("400x300")
    
    tk.Label(root, text="Hệ Thống Nhận Diện Khuôn Mặt", font=("Arial", 14)).pack(pady=20)
    tk.Button(root, text="Đăng nhập & Chụp ảnh", command=login, height=2, width=20).pack(pady=10)
    tk.Button(root, text="Nhận diện khuôn mặt", command=start_recognition, height=2, width=20).pack(pady=10)
    tk.Button(root, text="Thoát", command=root.quit, height=2, width=20).pack(pady=10)
    
    root.mainloop()

if __name__ == "__main__":
    main()
