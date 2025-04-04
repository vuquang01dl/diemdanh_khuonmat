import cv2
import numpy as np
import onnxruntime
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

# Khởi tạo FaceAnalysis
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0)  # Chạy trên CPU

EMBEDDINGS_PATH = r"D:\nhan_dien_khuon_mat_chinh_xac_cao\test_yolo_arcFce\data\faces-embeddings.npz"

def load_database():
    """ Load embeddings đã đăng ký """
    data = np.load(EMBEDDINGS_PATH, allow_pickle=True)
    names, embeddings = list(data.keys()), np.array(list(data.values()))
    return names, embeddings

def extract_embedding(frame):
    """ Trích xuất embedding từ frame camera """
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = app.get(img)
    
    if len(faces) == 0:
        return None, None

    return [face.normed_embedding for face in faces], [face.bbox for face in faces]

def recognize_face(frame, names, embeddings):
    """ Nhận diện tất cả khuôn mặt trong frame """
    face_embeddings, bboxes = extract_embedding(frame)
    
    if face_embeddings is None:
        return []

    results = []
    for emb, bbox in zip(face_embeddings, bboxes):
        similarities = cosine_similarity([emb], embeddings)[0]
        best_match_idx = np.argmax(similarities)
        best_score = similarities[best_match_idx]

        if best_score > 0.6:  # Ngưỡng nhận diện
            results.append((names[best_match_idx], bbox))
        else:
            results.append(("Không xác định", bbox))

    return results

def main():
    cap = cv2.VideoCapture(0)
    names, embeddings = load_database()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = recognize_face(frame, names, embeddings)

        for name, bbox in faces:
            x1, y1, x2, y2 = map(int, bbox)
            color = (0, 255, 0) if name != "Không xác định" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
