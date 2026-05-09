# 라이브러리 호출 및 준비
import os
import numpy as np
import pandas as pd
from scipy import signal
from scipy.io import loadmat
import matplotlib.pyplot as plt
import shutil

# numpy : 수치 계산 라이브러리
# pandas : 데이터 표를 다루는 라이브러리. 파일에서 베어링 데이터 추출.
# SciPy : 신호 처리용 라이브러리. 주파수 분석 및 Rescaling.
# Matplotlib.pyplot : 데이터 시각화 라이브러리. 이미지 파일 생성.



# 설정

# 데이터 경로 설정
CWRU_DATA_PATH = './cwru'        # CWRU 데이터셋 경로
OUT_BASE_PATH = './spectrogram_for_learning'   # 스펙트로그램 저장할 경로
TEST_FILE_PATH = './test_sensor_to_server/CWRU' # 테스트용 원본 파일 위치
os.makedirs(TEST_FILE_PATH, exist_ok=True)

# 샘플링 속도 설정(원본 20KHz에서 12KHz로 변환)
SAMPLING_RATE = 12000           # 원본 주파수값
TARGET_RATE = 12000             # 변환할 주파수값

# 윈도우 및 오버랩
WINDOW_SIZE = 10240             # 데이터를 일정한 길이로 나눔 (기존 20480)
# OVERLAP = 5120                  # 윗줄과 더불어 50% 겹치게 하여 데이터 수 늘림.(데이터 증대)
NORMAL_STEP = 1024  # 정상 데이터는 겹치지 않게 추출 (Step = Window)
FAULT_STEP = 256    # 고장 데이터는 75% 겹쳐서 추출 (데이터 수 4배 증가)

np.random.seed(0)

# 라벨링
LABELS = {
    'normal': 'Normal',
    'inner_race': 'IR',   # Inner Race Fault
    'outer_race': 'OR',   # Outer Race Fault
    'ball': 'B'        # Ball Fault
}



# 데이터 변환 함수
def save_spec(data, path):
    f, t, Sxx = signal.spectrogram(data, fs=TARGET_RATE)

    plt.figure(figsize=(2.24, 2.24))
    plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()



# 폴더 생성
for s in ['train', 'val', 'test']:
    for l in LABELS.keys():
        os.makedirs(os.path.join(OUT_BASE_PATH, s, l), exist_ok=True)

# CWRU 데이터 처리 로직
print("CWRU 데이터셋 변환 시작...")

# CWRU 폴더 내의 파일들을 읽어서 라벨별 분류 및 이미지 생성
for label_name, keyword in LABELS.items():
    # 해당 키워드를 포함하는 파일 리스트 추출
    target_files = [f for f in os.listdir(CWRU_DATA_PATH) if keyword in f and f.endswith('.mat')]

    for fname in target_files:
        try:
            # .mat 파일 로드 (CWRU 공식 데이터 기준)
            mat_data = loadmat(os.path.join(CWRU_DATA_PATH, fname))

            # 드라이브 엔드(DE) 또는 팬 엔드(FE) 진동 데이터 추출
            # CWRU 파일 내부의 Key값은 보통 'X097_DE_time' 같은 형식을 띰
            keys = [k for k in mat_data.keys() if 'DE_time' in k]
            if not keys: continue

            data = mat_data[keys[0]].flatten()

            # 클래스별로 Step size를 다르게 적용하여 비율을 맞춤
            if label_name == 'normal':
                step = NORMAL_STEP
            else:
                step = FAULT_STEP  # 고장 데이터는 더 많이 생성

            num_images = 0
            for start_idx in range(0, len(data) - WINDOW_SIZE + 1, step):
                sub_data = data[start_idx: start_idx + WINDOW_SIZE]

                # 데이터 분할 비율 결정 (8:1:1)
                rand_val = np.random.rand()        # 0으로 시드 고정.
                if rand_val < 0.8:
                    split = 'train'
                elif rand_val < 0.9:
                    split = 'val'
                else:
                    split = 'test'
                    # [추가] 테스트로 분류된 경우 원본 파일 복사
                    label_test_path = os.path.join(TEST_FILE_PATH, label_name)
                    os.makedirs(label_test_path, exist_ok=True)
                    shutil.copy2(os.path.join(CWRU_DATA_PATH, fname), os.path.join(label_test_path, fname))
                    ###

                save_path = os.path.join(OUT_BASE_PATH, split, label_name, f"{fname}_{num_images}.png")
                save_spec(sub_data, save_path)
                num_images += 1
                if num_images >= 150: break

            print(f"  - {fname}: {num_images}개 이미지 생성 완료")

        except Exception as e:
            print(f"  - {fname} 처리 중 오류: {e}")

print("완료!")