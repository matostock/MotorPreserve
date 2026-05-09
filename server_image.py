import numpy as np
import socket
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import os
import matplotlib

# GUI 없이 파일 저장만 하기 위해 Agg 백엔드 사용
matplotlib.use('Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

# AI 모델 설정 (server_v3.py 및 test_dataset_merged.py 참조)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs, 4)  # class_names: ball, inner_race, normal, outer_race
)
model.load_state_dict(torch.load('bearing_model.pth', map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# 전처리 설정
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class_names = ['ball', 'inner_race', 'normal', 'outer_race']

# 통계 변수 초기화 (test_dataset_merged.py 로직)
correct = 0
total = 0
class_correct = {name: 0 for name in class_names}
class_total = {name: 0 for name in class_names}


# 분석 결과 저장 로직 (server_v3.py 참조)
def save_analysis_result(img_pil, filename, label, pred_status):
    """이상 징후 발생 시 분석용 이미지 저장"""
    os.makedirs("./analysis_results", exist_ok=True)
    save_path = f"./analysis_results/{filename}_{pred_status}_target_{label}.png"

    # PIL 이미지를 다시 저장 (또는 시각적 처리가 필요하면 Figure 사용)
    img_pil.save(save_path)


# 서버 실행
PORT = 9000
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(('0.0.0.0', PORT))
    s.listen()
    print(f"통합 서버 가동 중 (Port: {PORT})... 데이터를 기다립니다.")

    while True:
        conn, addr = s.accept()
        with conn:
            data = b""
            while True:
                chunk = conn.recv(4096)
                if not chunk: break
                data += chunk
            if not data: continue

            try:
                # 헤더 파싱 (IMG_S:파일명:정답)
                header_line, content = data.split(b'\n', 1)
                header = header_line.decode('utf-8')
                parts = header.split(':')

                if parts[0] == "IMG_S":
                    fname = parts[1]
                    label = parts[2]

                    # 이미지 로드 및 추론
                    img = Image.open(io.BytesIO(content)).convert('RGB')
                    img_t = transform(img).unsqueeze(0).to(DEVICE)

                    with torch.no_grad():
                        outputs = model(img_t)
                        _, preds = torch.max(outputs, 1)
                        pred_status = class_names[preds[0]]

                    # 통계 업데이트
                    total += 1
                    class_total[label] += 1
                    is_hit = (pred_status == label)
                    if is_hit:
                        correct += 1
                        class_correct[label] += 1

                    color = "\33[94m" if is_hit else "\33[91m"
                    result_mark = "O" if is_hit else "X"
                    accuracy = 100 * correct / total
                    print(
                        f"[{color}{result_mark}\033[0m] 파일: {fname:30} | 실제: {label:10} | 예측: {pred_status:10} | 누적 정확도: {accuracy:.2f}%")

                    # 이상 징후 분석 저장 (정상이 아닌데 예측이 틀렸거나, 고장으로 판정된 경우)
                    if pred_status != 'normal':
                        save_analysis_result(img, fname, label, pred_status)

            except Exception as e:
                print(f"데이터 처리 오류: {e}")