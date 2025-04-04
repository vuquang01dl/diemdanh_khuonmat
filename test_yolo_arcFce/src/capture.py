import cv2
import os

def capture_images(save_dir="captured_images"):
    # Tạo thư mục lưu ảnh nếu chưa có
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Mở webcam
    cap = cv2.VideoCapture(0)
    
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không thể đọc từ webcam!")
            break
        
        # Hiển thị video
        cv2.imshow("Image Capture", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s'):
            # Resize ảnh về 640x640
            frame_resized = cv2.resize(frame, (640, 640))
            file_path = os.path.join(save_dir, f"image_{count+1}.jpg")
            cv2.imwrite(file_path, frame_resized)
            count += 1
            print(f"Đã lưu: {file_path}")
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Chụp ảnh hoàn tất!")

# Chạy hàm để chụp ảnh khi nhấn 's'
capture_images()
