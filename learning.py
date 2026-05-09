# 0. SSL 인증서 확인 과정을 무시하도록 설정 (에러 해결용)
import ssl
ssl._create_default_https_context = ssl._create_unverified_context



# 1. 라이브러리 호출 및 준비
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import sys
from tqdm import tqdm

# torch : Pytorch. 숫자 데이터를 텐서(Tensor)로 다루고 복잡한 연산 수행.
# torchvision : 이미지 관리를 위한 라이브러리



# 2. 학습 및 경로 설정 : 사용자가 직접 설정해야 한다. 모델의 학습에 큰 영향.
BATCH_SIZE = 16             # 학습 batch 크기 결정.
EPOCHS = 50                 # epoch 횟수 설정.
LEARNING_RATE = 0.0002      # 학습률. 가중치 업데이트 척도.
PATIENCE = 7                # 조기종료 시 손실이 줄어들지 않을 때 끝나지 않고 넘어가는 횟수

DATA_PATH = './spectrogram_for_learning'  # 이미지 데이터 폴더 경로
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # 학습 도구 설정.



# 3. 데이터 증강(Augmentation) 및 전처리
data_transforms = {
    'train': transforms.Compose([               # transforms.Compose : 여러 변환을 하나로 묶는 함수. 작성 순서대로 실행한다.
        transforms.Resize((224, 224)),          # 이미지의 크기를 224 x 224로 조정
        transforms.RandomHorizontalFlip(),      # 좌우 반전 추가. 데이터 증강 위함.
        transforms.RandomRotation(10),          # 미세한 회전 추가. CWRU와 IMS의 차이 극복 위한 데이터 증강.
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # 이미지의 밝기와 대비를 20% 범위 내에서 무작위 조정
        transforms.ToTensor(),                  # 이미지를 텐서로 변환(계산 가능한 형태로 변환)
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])      # 이미지의 RGP 평균과 표준편차를 사용해 정규화
    ]),
    'val': transforms.Compose([                 # 여기는 중간 점검용 이미지 전처리.
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
    # data_transforms : Dictionary
    # transforms.Compose : Container Function
    # 위에서의 방식으로 정규화 하는 이유 : 현재 모델인 resnet18이 학습한 데이터 형식과 통일 위해서



# 4. 학습할 데이터 로드

image_datasets = {}     # 데이터셋 객체 딕셔너리
dataloaders = {}        # 데이터 로더 딕셔너리

for i in ['train', 'val']:
    path = os.path.join(DATA_PATH, i)                               # 경로 생성
    dataset = datasets.ImageFolder(path, data_transforms[i])        # 데이터셋 객체 생성
    image_datasets[i] = dataset                                     # 'train'을 Key로 하여 저장.


for i in ['train', 'val']:
    loader = DataLoader(image_datasets[i], batch_size=BATCH_SIZE, shuffle=True) # 이미지 로더 생성.
    dataloaders[i] = loader         # 딕셔너리에 결과 저장.



# 5. 모델 설계
model = models.resnet18(weights='DEFAULT')
        # models.resnet18 클래스 호출.

# 전이 학습(필요에 따라 주석 해제)
# for param in model.parameters():
#     param.requires_grad = False         # 모델의 파라미터에 대해서 계산을 비활성화. 모델의 앞부분 층을 고정하여 전이 학습.

num_ftrs = model.fc.in_features         # 입력 피쳐 수 : 모델의 마지막 출력층 fc에 들어오는 입력 노드의 갯수 호출

# model.fc = nn.Linear(num_ftrs, 3)

model.fc = nn.Sequential(
    nn.Dropout(0.5),            # Dropout으로 뉴런의 50% 랜덤 비활성화. 노드간 의존도를 낮추어 과적합 방지.
    nn.Linear(num_ftrs, 4)      # 4개의 노드 출력하여 분류 수행(normal, inner_race, outer_race, ball)
)
model = model.to(DEVICE)        # 생성한 모델 인스턴스를 메모리로 전송.



# 6. 손실함수 및 최적화 도구
# criterion = nn.CrossEntropyLoss()                               # 손실 함수 인스턴스

# 클래스 순서: 0: ball, 1: inner_race, 2: normal, 3: outer_race
# 계산 근거: 최대 개수(3038) / 각 클래스 개수
weights = torch.tensor([1.0, 1.0, 3.5, 2.1], dtype=torch.float).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=weights)                 # 손실 함수 인스턴스
optimizer = optim.Adam(model.fc.parameters(), lr=0.0001)        # 최적화 인스턴스
        # CrossEntropyLoss : 손실함수. 예측값과 실제 정답 사이의 오차를 최소화.
        # model.fc.parameters : 모델에서 분류기 레이어만 학습하도록 범위 제한. 최종 출력층(fc)에 속한 파라미터만 다루도록 지정.
        # lr = Learning Rate.



# 7. 학습 시작
best_loss = float('inf') # 가장 낮은 손실을 저장 (무한대로 초기화)
counter = 0              # 손실이 줄어들지 않은 횟수

print(f"학습 시작 (Device: {DEVICE})...")

for epoch in range(EPOCHS):
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0

        # for inputs, labels in dataloaders[phase]:
        #     inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        pbar = tqdm(dataloaders[phase], unit="batch", file = sys.stdout)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            # 진행 표시바 옆에 실시간 Loss 표시 (선택 사항)
            pbar.set_description(f"Epoch {epoch + 1}/{EPOCHS} [{phase}]")
            pbar.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(image_datasets[phase])
        epoch_acc = running_corrects.double() / len(image_datasets[phase])

        print(f'Epoch {epoch+1}/{EPOCHS} {phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # --- Early Stopping 로직 추가 ---
        if phase == 'val':
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                counter = 0  # 손실이 갱신되면 카운터 리셋
                # 가장 좋은 성능의 모델 저장
                torch.save(model.state_dict(), "bearing_model.pth")
                print(f"--> 모델 저장 완료 (Loss: {best_loss:.4f})")
            else:
                counter += 1  # 손실이 줄어들지 않으면 카운터 증가
                print(f"--> EarlyStopping 카운터: {counter} / {PATIENCE}")

            if counter >= PATIENCE:
                print("Early Stopping 발생! 학습을 종료합니다.")
                break  # val 루프 탈출

    if counter >= PATIENCE:
        break  # epoch 루프 탈출