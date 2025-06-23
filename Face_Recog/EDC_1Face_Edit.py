import cv2
import torch
import numpy as np
from torchvision.models import resnet50, ResNet50_Weights
from torch import nn
import torch.nn.functional as F
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import time
import json
from collections import defaultdict

# Thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MTCNN và nhận diện khuôn mặt
mtcnn = MTCNN(keep_all=True, device=device)
face_recog_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
known_embeddings, known_names = torch.load(r"D:\Code\y4_semester2\AI_Project\Model\Final_Model\new_face_db.pth", map_location=device)

# Mô hình cảm xúc
emotion_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
emotion_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
num_ftrs = emotion_model.fc.in_features
emotion_model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 7)
)
checkpoint = torch.load(r"D:\Code\y4_semester2\AI_Project\Model\Final_Model\4_Model_checkpoint.pth", map_location=device, weights_only=True)
emotion_model.load_state_dict(checkpoint['model_state_dict'])
emotion_model = emotion_model.to(device)
emotion_model.eval()

FER2013_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def preprocess_emotion_face(face_gray):
    face_gray = cv2.resize(face_gray, (48, 48), interpolation=cv2.INTER_AREA)
    face_tensor = torch.from_numpy(face_gray).unsqueeze(0).unsqueeze(0).float()
    face_tensor = (face_tensor - 127.5) / 127.5
    return face_tensor.to(device)

# Khởi tạo biến
start_time = time.time()
last_time = start_time
person_emotion_stats = defaultdict(lambda: {
    emotion: {"score_sum": 0.0, "duration": 0.0, "count": 0} for emotion in FER2013_LABELS
})

# video_path = r"D:\Code\y4_semester2\AI_Project\Model\Final_Model\new_face_db.pth"
# cap = cv2.VideoCapture(video_path)
cap = cv2.VideoCapture(0)

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb_frame)
    boxes, _ = mtcnn.detect(img)
    faces = mtcnn.extract(img, boxes, save_path=None) if boxes is not None else []

    for i, face_tensor in enumerate(faces):
        name = "Unknown"
        min_dist = 1.0

        # Nhận diện tên
        with torch.no_grad():
            emb = face_recog_model(face_tensor.unsqueeze(0).to(device))
        for db_emb, db_name in zip(known_embeddings, known_names):
            dist = (emb - db_emb).norm().item()
            if dist < min_dist and dist < 0.9:
                min_dist = dist
                name = db_name

        # Cắt khuôn mặt
        x1, y1, x2, y2 = [int(coord) for coord in boxes[i]]
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue

        gray_face = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)

        try:
            input_face = preprocess_emotion_face(gray_face)
            with torch.no_grad():
                output = emotion_model(input_face)
                prob = F.softmax(output, dim=1)
                conf, pred = torch.max(prob, 1)
                label = FER2013_LABELS[pred.item()]
                confidence = conf.item()
        except:
            continue

        # Cập nhật thống kê theo người
        current_time = time.time()
        delta_time = current_time - last_time
        last_time = current_time

        person_stats = person_emotion_stats[name]
        person_stats[label]["score_sum"] += confidence * delta_time
        person_stats[label]["duration"] += delta_time
        person_stats[label]["count"] += 1

        # Hiển thị
        display_text = f"{name} - {label} ({confidence:.2f})"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, display_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)

    cv2.imshow("Emotion + Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Ghi file JSON
output_json = []
for person_name, emotions in person_emotion_stats.items():
    emotion_info = {}
    for emotion, stats in emotions.items():
        if stats["duration"] > 0:
            avg_score = stats["score_sum"] / stats["duration"]
            emotion_info[emotion] = {
                "score": round(avg_score, 2),
                "duration": round(stats["duration"], 2)
            }
    output_json.append({
        "name": person_name,
        "emotions": emotion_info
    })

with open("emotion_per_person.json", "w") as f:
    json.dump(output_json, f, indent=4)

# In kết quả
print("\nKết quả từng người:")
for person in output_json:
    print(f"\n👤 {person['name']}")
    for emotion, data in person["emotions"].items():
        print(f"  {emotion}: Score: {data['score']:.2f}, Duration: {data['duration']:.2f}s")
