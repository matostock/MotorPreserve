import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# 1. 설정
DATA_PATH = './spectrogram_for_learning/test'
MODEL_PATH = 'bearing_model.pth'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 데이터 전처리 (학습 때와 동일하게)
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. 데이터 로더
test_dataset = datasets.ImageFolder(DATA_PATH, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
class_names = test_dataset.classes  # ['normal', 'ball', 'inner_race', 'outer_race']

# 4. 모델 불러오기
model = models.resnet18()
num_ftrs = model.fc.in_features
# 학습 때 dropout을 했으므로 여기도 추가.
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs, 4)
)
# model.load_state_dict(torch.load(MODEL_PATH))                           # 모델 가중치를 불러올 시 저장 당시의 장치 정보를 따름
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))      # 모델 가중치를 불러올 시 현재 환경에 맞게 재배치
model = model.to(DEVICE)
model.eval()

# 5. 테스트 시작
print(f"테스트 시작 (대상: {len(test_dataset)}개 파일)...")

correct = 0
total = 0

# 각 클래스별 맞춘 개수를 확인하기 위한 변수
class_correct = {name: 0 for name in class_names}
class_total = {name: 0 for name in class_names}

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        # if preds == labels:
        #     correct += 1

        # 결과 집계
        if preds[0] == labels[0]:
            correct += 1
            class_correct[class_names[labels[0]]] += 1
        class_total[class_names[labels[0]]] += 1
        total += 1

        # 개별 결과 출력 (선택 사항)
        print(f"실제: {class_names[labels[0]]} | 예측: {class_names[preds[0]]}")

# 6. 결과 출력
print("-" * 30)
print(f"전체 정확도(Accuracy): {100 * correct / total:.2f}%")
print("-" * 30)
print("클래스별 세부 정확도:")
for name in class_names:
    if class_total[name] > 0:
        acc = 100 * class_correct[name] / class_total[name]
        print(f" - {name:10}: {acc:.2f}% ({class_correct[name]}/{class_total[name]})")
    else:
        print(f" - {name:10}: 데이터 없음")