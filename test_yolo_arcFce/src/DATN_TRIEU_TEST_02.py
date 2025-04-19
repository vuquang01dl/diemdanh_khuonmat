import cv2
import os
import numpy as np
import onnxruntime
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
from tkinter import messagebox, simpledialog
from PIL import Image, ImageTk
import time
from ultralytics import YOLO
import pygame  # Thêm pygame

# Khởi tạo pygame
pygame.mixer.init()

# Đường dẫn âm thanh cảnh báo
ALERT_SOUND_PATH = r"C:\Users\V5030587\Downloads\pypj\Thang_may_nhan_dien_khuon_mat\Thang_may_nhan_dien_khuon_mat\test_yolo_arcFce\src\canhbao.mp3"

# Đường dẫn lưu embeddings
EMBEDDINGS_PATH = r"D:\nhan_dien_khuon_mat_chinh_xac_cao\test_yolo_arcFce\data\faces-embeddings.npz"
SAVE_DIR = r"D:\\nhan_dien_khuon_mat_chinh_xac_cao\\test_yolo_arcFce\\data\\registered_faces"

yolo_model = YOLO("yolov8n.pt")
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0)

def load_database():
    if not os.path.exists(EMBEDDINGS_PATH):
        return [], np.array([])
    data = np.load(EMBEDDINGS_PATH, allow_pickle=True)
    return list(data.keys()), np.array(list(data.values()))

def remove_embeddings():
    username = simpledialog.askstring("Xóa người dùng", "Nhập tên người dùng cần xóa:")
    if not username:
        return
    if not os.path.exists(EMBEDDINGS_PATH):
        messagebox.showerror("Lỗi", "Không tìm thấy dữ liệu embeddings!")
        return
    data = np.load(EMBEDDINGS_PATH, allow_pickle=True)
    data_dict = dict(data)
    if username in data_dict:
        data_dict.pop(username, None)
        np.savez(EMBEDDINGS_PATH, **data_dict)
        messagebox.showinfo("Thành công", f"Đã xóa ID: {username}")
    else:
        messagebox.showerror("Lỗi", "Không tìm thấy ID trong hệ thống!")

def capture_images(username):
    os.makedirs(SAVE_DIR, exist_ok=True)
    cap = cv2.VideoCapture(0)
    count = 0
    embeddings = {}
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Capture Image", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            file_path = os.path.join(SAVE_DIR, f"{username}_{count}.jpg")
            cv2.imwrite(file_path, frame)
            embedding, _ = extract_embedding(frame)
            if embedding:
                embeddings[f"{username}_{count}"] = embedding[0]
            count += 1
        elif key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    if embeddings:
        save_embeddings(embeddings)

def extract_embedding(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_app.get(img)
    if not faces:
        return None, None
    return [face.normed_embedding for face in faces], [face.bbox for face in faces]

def save_embeddings(new_embeddings):
    names, embeddings = load_database()
    names.extend(new_embeddings.keys())
    embeddings = np.vstack([embeddings, list(new_embeddings.values())]) if embeddings.size else np.array(list(new_embeddings.values()))
    np.savez(EMBEDDINGS_PATH, **dict(zip(names, embeddings)))

def recognize_face(frame, names, embeddings):
    face_embeddings, bboxes = extract_embedding(frame)
    if face_embeddings is None or len(face_embeddings) == 0:
        return []

    results = []
    for emb, bbox in zip(face_embeddings, bboxes):
        if embeddings.size == 0:
            name = "Unknown"
        else:
            similarities = cosine_similarity([emb], embeddings)[0]
            best_match_idx = np.argmax(similarities)
            best_score = similarities[best_match_idx]
            name = names[best_match_idx] if best_score > 0.6 else "Unknown"
        results.append((name, bbox))
    return results

def start_recognition():
    names, embeddings = load_database()
    cap = cv2.VideoCapture(0)

    unknown_start_time = None
    unknown_detected = False
    alert_sent = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo_model(frame)[0]
        person_boxes = []
        for r in results.boxes.data:
            cls_id = int(r[5])
            if cls_id == 0:
                x1, y1, x2, y2 = map(int, r[:4])
                person_boxes.append((x1, y1, x2, y2))

        faces = []
        for (x1, y1, x2, y2) in person_boxes:
            person_crop = frame[y1:y2, x1:x2]
            sub_faces = recognize_face(person_crop, names, embeddings)
            for name, bbox in sub_faces:
                fx1, fy1, fx2, fy2 = map(int, bbox)
                abs_bbox = (x1 + fx1, y1 + fy1, x1 + fx2, y1 + fy2)
                faces.append((name, abs_bbox))

        unknown_present = any(name == "Unknown" for name, _ in faces)

        if unknown_present:
            if not unknown_detected:
                unknown_start_time = time.time()
                unknown_detected = True
                alert_sent = False
            else:
                elapsed = time.time() - unknown_start_time
                if elapsed >= 10 and not alert_sent:
                    print("Đã gửi tin nhắn về hệ thống phát hiện người lạ")
                    pygame.mixer.music.load(ALERT_SOUND_PATH)
                    pygame.mixer.music.play()
                    alert_sent = True
        else:
            unknown_detected = False
            unknown_start_time = None
            alert_sent = False

        for name, bbox in faces:
            x1, y1, x2, y2 = bbox
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("Face Recognition with YOLO", frame)
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
    root.geometry("400x350")
    
    tk.Label(root, text="Hệ Thống Nhận Diện Khuôn Mặt", font=("Arial", 14)).pack(pady=20)
    tk.Button(root, text="Đăng nhập & Chụp ảnh", command=login, height=2, width=20).pack(pady=10)
    tk.Button(root, text="Nhận diện khuôn mặt", command=start_recognition, height=2, width=20).pack(pady=10)
    tk.Button(root, text="Xóa người dùng", command=remove_embeddings, height=2, width=20).pack(pady=10)
    tk.Button(root, text="Thoát", command=root.quit, height=2, width=20).pack(pady=10)
    
    root.mainloop()

if __name__ == "__main__":
    main()
