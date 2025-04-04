import cv2
import os
import numpy as np
import onnxruntime
import smtplib
import RPi.GPIO as GPIO
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

# Thiết lập GPIO
RELAY_PIN = 13  # GPIO13 để điều khiển relay
BUTTON_PIN = 2  # GPIO2 để chụp ảnh và gửi email
GPIO.setmode(GPIO.BCM)
GPIO.setup(RELAY_PIN, GPIO.OUT)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

# Đường dẫn dữ liệu
EMBEDDINGS_PATH = "/home/pi/face_data/faces-embeddings.npz"
SAVE_DIR = "/home/pi/face_data/registered_faces"
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0)

# Load dữ liệu embeddings
def load_database():
    if not os.path.exists(EMBEDDINGS_PATH):
        return [], np.array([])
    data = np.load(EMBEDDINGS_PATH, allow_pickle=True)
    return list(data.keys()), np.array(list(data.values()))

# Nhận diện khuôn mặt
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

# Trích xuất embedding
def extract_embedding(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = app.get(img)
    if not faces:
        return None, None
    return [face.normed_embedding for face in faces], [face.bbox for face in faces]

# Mở cửa nếu khuôn mặt tồn tại trong database
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
            if name != "Unknown":
                GPIO.output(RELAY_PIN, GPIO.HIGH)
                time.sleep(10)  # Mở cửa trong 10 giây
                GPIO.output(RELAY_PIN, GPIO.LOW)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0) if name != "Unknown" else (0, 0, 255), 2)
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Gửi email khi phát hiện nút nhấn
def send_email_with_attachment(image_path):
    sender_email = "vuquang01dl@gmail.com"
    sender_app_password = "wgmc xccj soba owzz"
    receiver_email = "vuquangclone01@gmail.com"
    
    subject = "Cảnh báo: Phát hiện người nhấn nút!"
    body = "Có người nhấn nút tại thang máy. Xem ảnh đính kèm."
    
    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))
    
    with open(image_path, "rb") as attachment:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment.read())
    encoders.encode_base64(part)
    part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(image_path)}")
    msg.attach(part)
    
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_app_password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
        print("Email sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {e}")

# Giám sát nút nhấn để chụp ảnh
def monitor_button():
    cap = cv2.VideoCapture(0)
    while True:
        if GPIO.input(BUTTON_PIN) == GPIO.HIGH:
            print("Button pressed! Capturing image...")
            ret, frame = cap.read()
            if ret:
                image_path = "/home/pi/face_data/alert.jpg"
                cv2.imwrite(image_path, frame)
                send_email_with_attachment(image_path)
                time.sleep(2)  # Tránh gửi nhiều email liên tục
    cap.release()

if __name__ == "__main__":
    try:
        start_recognition()
        monitor_button()
    except KeyboardInterrupt:
        print("Stopping system...")
        GPIO.cleanup()
