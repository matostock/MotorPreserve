import numpy as np
import socket
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from scipy import signal
import io
import matplotlib
matplotlib.use('Agg')  # GUI 백엔드를 사용하지 않도록 강제 설정 (백색 화면 방지 핵심)
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

# 모델 및 설정 로드
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs, 4)
)
model.load_state_dict(torch.load('bearing_model.pth', map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class_names = ['ball', 'inner_race', 'normal', 'outer_race']

# 통계 관리를 위한 변수 (test_dataset_merged 로직 적용)
correct = 0
total = 0
class_correct = {name: 0 for name in class_names}
class_total = {name: 0 for name in class_names}

def predict(content, fs_input, label):
    bearing_data = np.frombuffer(content, dtype=np.float64)
    TARGET_FS = 12000

    if int(fs_input) != TARGET_FS:
        num_samples = int(len(bearing_data) * TARGET_FS / int(fs_input))
        bearing_data = signal.resample(bearing_data, num_samples)

    # 기존 스펙트로그램 생성 로직 복구
    f, t, Sxx = signal.spectrogram(bearing_data, fs=TARGET_FS)

    # matplotlib를 사용하되 화면 출력 없이 메모리 내에서만 작업
    fig = Figure(figsize=(2.24, 2.24))
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
    ax.axis('off')

    # 메모리 버퍼에 저장
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)

    # 이미지 변환 및 추론
    img = Image.open(buf).convert('RGB')
    img_t = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_t)
        _, preds = torch.max(outputs, 1)
        pred_status = class_names[preds[0]]

    # 사용한 객체와 버퍼를 즉시 닫아 메모리 및 GUI 자원 해제
    plt.close(fig) # Figure 객체 닫기
    buf.close()    # 버퍼 닫기

    return pred_status

# 서버 실행 및 실시간 통계 출력
PORT = 9000
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(('0.0.0.0', PORT))
    s.listen()
    print("서버 가동 중... 실시간 통계를 기록합니다.")

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
                header_line, content = data.split(b'\n', 1)
                header = header_line.decode('utf-8')
                dtype, fname, b_no, label, source, fs_input = header.split(':')

                if dtype == "RAW_S":
                    pred_status = predict(content, fs_input, label)

                    # 결과 집계 및 출력
                    total += 1
                    class_total[label] += 1

                    is_hit = (pred_status == label)
                    if is_hit:
                        correct += 1
                        class_correct[label] += 1

                    result_mark = "O" if is_hit else "X"

                    # 실시간 요약 출력
                    print(
                        f"[{result_mark}] 실제: {label:10} | 예측: {pred_status:10} | 전체 정확도: {100 * correct / total:5.2f}%")

            except Exception as e:
                print(f"데이터 처리 오류: {e}")