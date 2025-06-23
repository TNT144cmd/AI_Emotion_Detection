import cv2
from cv2 import CascadeClassifier
import torch
import numpy as np
from torchvision.models import resnet50, ResNet50_Weights
from torch import nn
import torch.nn.functional as F
import time
import json

# Load Haar Cascade classifier
face_cascade = CascadeClassifier(r"D:\Code\y4_semester1\Face_detection\haarcascade\haarcascade_frontalface_default.xml")
if face_cascade.empty():
    raise ValueError("Error loading Haar Cascade classifier. Check file path.")

# 1. Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tải ResNet-50 với weights mới
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 7)
)

# Load checkpoint
checkpoint_path = r"D:\Code\y4_semester2\AI_Project\Model\4_Model_checkpoint.pth"
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

# Định nghĩa tiền xử lý trực tiếp từ NumPy
mean = 0.5
std = 0.5

def preprocess_face(face):
    # Resize
    face = cv2.resize(face, (48, 48), interpolation=cv2.INTER_AREA)
    # Chuyển sang tensor, thêm chiều channel và batch
    face_tensor = torch.from_numpy(face).unsqueeze(0).unsqueeze(0).float()  # Shape: [1, 1, 48, 48]
    # Normalize
    face_tensor = (face_tensor - mean * 255) / (std * 255)
    return face_tensor.to(device)

# Ánh xạ nhãn cảm xúc
FER2013_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Khởi tạo thời gian bắt đầu
start_time = time.time()

# Lưu tổng thời gian và confidence dựa trên thời gian cho mỗi cảm xúc
emotion_durations = {emotion: 0.0 for emotion in FER2013_LABELS}  # Tổng thời gian
emotion_confidence_time = {emotion: {"total_conf_time": 0.0, "total_time": 0.0} for emotion in FER2013_LABELS}  # Tổng tích confidence và thời gian
current_emotion = None
start_emotion_time = None
last_time = start_time  # Thời điểm của lần lặp trước

# Mở webcam
#video_path = r"D:\Code\y4_semester2\AI_Project\TestCase\IMG_8530.MOV"
video_path = r"D:\Code\y4_semester2\AI_Project\TestCase\BaoAn_My_Hoc.MOV"

cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame or end of video.")
        break

    # Chuyển frame sang grayscale để phát hiện khuôn mặt
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Phát hiện khuôn mặt
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    print(f"Detected faces: {len(faces)}")

    # Xử lý tất cả khuôn mặt được phát hiện (lấy khuôn mặt đầu tiên nếu có)
    if len(faces) > 0:
        (x, y, w, h) = faces[0]  # Lấy khuôn mặt đầu tiên trong danh sách

        # Kiểm tra kích thước vùng khuôn mặt
        if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > gray.shape[1] or y + h > gray.shape[0]:
            print(f"Invalid face region at ({x}, {y}, {w}, {h}). Skipping.")
            continue

        face = gray[y:y+h, x:x+w].copy()

        try:
            # Debug thông tin về face
            print(f"Face shape: {face.shape}, dtype: {face.dtype}, min: {face.min()}, max: {face.max()}")

            # Chuẩn hóa giá trị pixel
            if face.max() > 255 or face.min() < 0:
                face = np.clip(face, 0, 255).astype(np.uint8)

            # Tiền xử lý trực tiếp
            face_tensor = preprocess_face(face)

            # Dự đoán cảm xúc
            with torch.no_grad():
                outputs = model(face_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                label = FER2013_LABELS[predicted.item()]
                confidence = confidence.item()

            # Tính thời gian và cập nhật
            current_time = time.time()
            delta_time = current_time - last_time
            last_time = current_time

            if delta_time > 0:  # Đảm bảo delta_time hợp lệ
                if current_emotion is not None:
                    # Cập nhật thời gian và confidence cho cảm xúc trước
                    emotion_durations[current_emotion] += delta_time
                    emotion_confidence_time[current_emotion]["total_conf_time"] += confidence * delta_time
                    emotion_confidence_time[current_emotion]["total_time"] += delta_time

                current_emotion = label
                if start_emotion_time is None:
                    start_emotion_time = current_time
                print(f"Current Emotion: {label}, Running Duration: {current_time - start_emotion_time:.2f}s, Confidence: {confidence:.2f}")

                # Vẽ nhãn và hình chữ nhật
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"Kha Ngan - {label} ({(current_time - start_emotion_time)/60:.2f}min, {confidence:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        except Exception as e:
            print(f"Error processing face at ({x}, {y}): {str(e)}")

    # Hiển thị frame
    cv2.imshow('Emotion Detection', frame)

    # Thoát khi nhấn 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cập nhật thời gian và confidence cuối cùng khi thoát
if current_emotion is not None:
    current_time = time.time()
    delta_time = current_time - last_time
    if delta_time > 0:
        emotion_durations[current_emotion] += delta_time
        emotion_confidence_time[current_emotion]["total_conf_time"] += confidence * delta_time
        emotion_confidence_time[current_emotion]["total_time"] += delta_time

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()

# Lưu dữ liệu vào file JSON, chỉ giữ các cảm xúc có thời gian > 0
output_data = {
    "start_time": time.ctime(start_time),
    "emotion_durations": {
        emotion: {
            "average_confidence": f"{emotion_confidence_time[emotion]['total_conf_time'] / emotion_confidence_time[emotion]['total_time']:.2f}" if emotion_confidence_time[emotion]['total_time'] > 0 else "N/A",
            "duration": f"{duration/60:.2f}min",
        } for emotion, duration in emotion_durations.items() if duration > 0
    }
}
with open("emotion_data.json", "w") as f:
    json.dump(output_data, f, indent=4)

# In kết quả thời gian và cảm xúc
print("\nTotal Duration and Average Confidence for Each Emotion (also saved to emotion_data.json):")
for emotion, duration in emotion_durations.items():
    if duration > 0:
        avg_conf = emotion_confidence_time[emotion]['total_conf_time'] / emotion_confidence_time[emotion]['total_time'] if emotion_confidence_time[emotion]['total_time'] > 0 else 0.0
        print(f"  {emotion}: Duration: {duration/60:.2f}min, Average Confidence: {avg_conf:.2f}")