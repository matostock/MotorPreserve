# 1. 라이브러리 호출 및 준비
import os
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
import shutil

# numpy : 수치 계산 라이브러리
# pandas : 데이터 표를 다루는 라이브러리. 파일에서 베어링 데이터 추출.
# SciPy : 신호 처리용 라이브러리. 주파수 분석 및 Rescaling.
# Matplotlib.pyplot : 데이터 시각화 라이브러리. 이미지 파일 생성.



# 2. 설정

# 데이터 경로 설정
RAW_DATA_PATH = './ism'    # 원본 데이터 위치
OUT_BASE_PATH = './spectrogram_for_learning'  # 변환한 이미지 파일을 저장할 장소
TEST_FILE_PATH = './test_sensor_to_server/ISM'
os.makedirs(TEST_FILE_PATH, exist_ok=True)

# 샘플링 속도 설정(원본 20KHz에서 12KHz로 변환)
SAMPLING_RATE = 20000           # 원본 주파수값
TARGET_RATE = 12000             # 변환할 주파수값

# 윈도우 및 오버랩
WINDOW_SIZE = 10240             # 데이터를 일정한 길이로 나눔 (기존 20480)
OVERLAP = 5120                  # 윗줄과 더불어 50% 겹치게 하여 데이터 수 늘림.(데이터 증대)

LABELS = {
    'normal': (0, 1500),
    'inner_race': (1500, 2156), # Bearing 3 데이터 사용 예정
    'ball': (1500, 2156)       # Bearing 4 데이터 사용 예정
}



# 3. 데이터 변환 함수
def save_spec(data, path):

    # Resampling : 데이터에서 샘플링 주파수 변경.
    num_samples = int(len(data) * TARGET_RATE / SAMPLING_RATE)      # 샘플 개수 줄임 : 속도 증가, 용량 절약, 노이즈 제거.
            # [기존 데이터 길이 len(data)] * [변환할 주파수값] / [기존 주파수값]
    resampled = signal.resample(data, num_samples)                  # 리셈플된 주파수 데이터.

    # STFT(Short-Time Fourier Transform) : 진동 데이터를 시간대별 주파수 성분으로 분해.
    f, t, Sxx = signal.spectrogram(resampled, fs=TARGET_RATE)
            # signal.spectrogram이 반환하는 값
            # f(frequency), t(time), Sxx(Spectrogram. S는 Signal, xx는 상관관계)

    # 이미지 저장 (불필요한 정보 제거하고 특징만 저장)
    plt.figure(figsize=(2.24, 2.24))  # CNN 입력 사이즈에 맞춤 (224x224 px)
    plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
            # plt.pcolormesh : 2차원 배열의 값을 색상 격차로 표현하는 함수.
                    # t : x축 값. 여기서는 시간.
                    # f : y축 값. 여기선 주파수.
                    # 10 * np.log10(Sxx + 1e-10) : 색상 농도. 자세히 알아둘 필요는 없을듯.
                    # shading : 색칠하기. 'flat'은 단색, 'gouraud'는 격자간 색상 보정해서 계단 효과 완화.(그라데이션?)
                    # cmap : Color Map. 'viridis'는 여러 색상 중 하나. 단순히 그것뿐.
    plt.axis('off') # 테두리와 눈금 제거
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
            # 실제 파일로 저장.
                    # bbox_inches : Bounding Box. 이미지 여백 설정.
                    # pad_inches : 이것도 여백. 조금 더 미세한 설정.
    plt.close()
            # 메모리 관리 위해 명시적 해제. plt.figure()의 경우 자동으로 메모리 관리를 할 수 없기 때문.



# 4. 폴더 생성
for s in ['train', 'val', 'test']:
    for l in LABELS.keys():
        os.makedirs(os.path.join(OUT_BASE_PATH, s, l), exist_ok=True)
                # os.makedirs : 폴더 생성 함수
                # os.path.join : 경로 표기 위해 문자열 합침. 구분자는 os에 맞게 자동으로 입력됨.
                        # s : 'train', 'val', 'test'
                        # l : 라벨값. 'normal', 'suspect', 'failure'
                # exist_ok : 폴더가 이미 있을 경우 계속 진행할지 여부.

all_files = sorted([f for f in os.listdir(RAW_DATA_PATH) if os.path.isfile(os.path.join(RAW_DATA_PATH, f))])
        # sorted : 인자로 들어온 리스트 요소들을 오름차순으로 정렬. 파일 번호 순서대로 데이터 처리.
                # os.listdir(RAW_DATA_PATH) : RAW_DATA_PATH 폴더 안에 있는 파일/폴더 이름을 리스트로 가져옴.
                # f : 임시 명칭.
                # if os.path.isfile(os.path.join(RAW_DATA_PATH, f)) : 폴더 내에 하위 폴더가 있는 경우 파일만 검출하는 필터링 조건식.



# 5. 데이터 배분 및 변환
for label, (start, end) in LABELS.items():
        # label, (start, end) : unpacking 방식. 위에서 정의한 LABELS에서의 값과 label의 번호를 대조하여 활용.(normal - suspect - failure 판단용)
        # .items : Dictionary에서 Key와 Value를 (Key, Value)형태로 반환.

    label_files = all_files[start:end + 1]


    # 정상/열화 데이터 추출 개수를 500개로 설정
    if label != 'failure':
        step = max(1, len(label_files) // 500)
                # 1500 / 500으로 step은 3이 된다. 샘플링 시 3번째마다 추출한다는 계산이 나옴.
                # max(1,)은 파일이 500개 이하인 경우 step이 0이 되는 것을 방지하기 위함.(최소 1개 보장)
        label_files = label_files[::step]
                # [::step] : 리스트에서 step에 해당하는 순서에서 샘플링.


    # 8:1:1 비율로 데이터 분할
    n = len(label_files)
    splits = {'train': label_files[:int(n * 0.8)], 'val': label_files[int(n * 0.8):int(n * 0.9)],
              'test': label_files[int(n * 0.9):]}

    print(f"[{label}] 데이터 생성 중...")


    # 데이터 로드 및 특정 열 추출
    for split_name, files in splits.items():

        # [추가] 현재 분할이 test인 경우 해당 라벨의 원본 파일들을 모두 복사
        if split_name == 'test':
            label_test_path = os.path.join(TEST_FILE_PATH, label)
            os.makedirs(label_test_path, exist_ok=True)
            for f_to_copy in files:
                shutil.copy2(os.path.join(RAW_DATA_PATH, f_to_copy), os.path.join(label_test_path, f_to_copy))
        ###

            # splits Dictionary에서 Key에 해당하는 'train', 'val', 'test' 파일명과 그에 속한 파일 이름 'List' files를 반환.
        for idx, fname in enumerate(files):
                # 이쪽 for에서는 앞선 for에서 가져온 List인 files를 다시 한 번 다룸. Dictionary가 아니므로 키/값이 아니라 순번/요소이다.
                # enumerate : List 내의 파일 이름(fname)과 그 번호(idx)를 반환하는 함수.
            try:
                df = pd.read_csv(os.path.join(RAW_DATA_PATH, fname), sep='\\s+', header=None)
                        # pd.reda_csv : 텍스트 파일을 읽어 표 형태로 만듦.
                            # sep = "\\s+" : 공백을 기준으로 나눔.
                # data = df.iloc[:, 4].values
                #         # Bearing 3 (내륜 결함) 데이터 사용
                #         # iloc : Inter Location. 표에서 정수 인덱스(번호)를 사용하여 행과 열을 선택하는 함수.

                # 라벨에 따른 베어링 열 선택
                if label == 'inner_race':
                    bearings_to_process = [(4, "b3")]  # 3번 베어링
                elif label == 'ball':
                    bearings_to_process = [(5, "b4")]  # 4번 베어링
                else:
                    bearings_to_process = [(2, "b1"), (3, "b2")]    # 1번, 2번 베어링

                # 결함 데이터의 경우 이미지 추가 생성(파일 하나당 여러 이미지를 생성하여 데이터 증강. Sliding Window.)
                for col_idx, b_name in bearings_to_process:
                    data = df.iloc[:, col_idx].values
                    if label in ['inner_race', 'ball']:
                        for i, start_idx in enumerate(range(0, len(data) - WINDOW_SIZE + 1, OVERLAP)):
                                # 0에서 시작하여 WINDOW_SIZE만큼 이동.
                                # OVERLAP만큼 이동하여 시작 지점(start_idx) 결정.
                            sub_data = data[start_idx: start_idx + WINDOW_SIZE]
                                    # 특정 구간 추출
                            save_path = os.path.join(OUT_BASE_PATH, split_name, label, f"{fname}_{i}.png")
                            save_spec(sub_data, save_path)
                    else:   # 결함 데이터가 아닌 경우
                        save_path = os.path.join(OUT_BASE_PATH, split_name, label, f"{fname}.png")
                        # save_spec(data, save_path)
                        # 원본 데이터가 WINDOW_SIZE보다 클 수 있으므로 안전하게 슬라이싱
                        save_spec(data[:WINDOW_SIZE], save_path)
            except:
                continue
    print(f"[{label}] 완료!")