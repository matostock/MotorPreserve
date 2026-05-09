import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# 1. 데이터 경로 설정
data_path = "./1st_test"

# files = sorted(os.listdir(data_path))

# 폴더 내 파일 확인
files = sorted([f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))])
if not files:
    print("경고: 폴더 내에 파일이 없습니다. 경로를 다시 확인하세요.")
else:
    print(f"총 {len(files)}개의 파일을 찾았습니다.")

rms_list = []

# 속도 향상을 위해 10개 파일마다 1개씩 분석 (샘플링)
sample_step = 10

print(f"분석 시작 (총 {len(files)}개 파일 중 {len(files)//sample_step}개 샘플링)...")

# # 2. 첫 번째 파일만 먼저 테스트 출력
# test_file = os.path.join(data_path, files[0])
# test_df = pd.read_csv(test_file, sep='\t', header=None)
# print("첫 번째 파일 데이터 샘플:")
# print(test_df.head()) # 데이터가 숫자로 잘 나오는지 확인

# 2. 모든 파일을 순회하며 RMS 계산 (데이터가 많으므로 샘플링해서 확인 가능)
# for file in files:
for i in range(0, len(files), sample_step):
    try:
        file_path = os.path.join(data_path, files[i])
        # # Test 1은 탭(\t)으로 구분된 8채널 데이터
        # df = pd.read_csv(file_path, sep='\t', header=None)
        # # sep='\t' 또는 sep='\s+' (공백 대응)

        # 파일이 비어있는지 확인
        if os.path.getsize(file_path) == 0:
            continue

        # 데이터 로드 (공백/탭 모두 대응)
        df = pd.read_csv(file_path, sep='\\s+', header=None)

        # 데이터가 충분한지 확인 (20480행이 맞는지)
        if df.empty or len(df) < 100:
            continue

        # 요구사항 반영: 수평 채널(0, 2, 4, 6번 인덱스)만 선택
        selected_channels = df[[0, 2, 4, 6]]

        # RMS 계산: sqrt(mean(x^2))
        rms = np.sqrt(np.mean(selected_channels ** 2, axis=0))
        # rms_values.append(rms)
        if not np.isnan(rms).any():
            rms_list.append(rms.values)

        if len(rms_list) % 20 == 0:
            print(f"진행 중: {i}/{len(files)} 파일 완료")

    except Exception as e:
        continue

# 3. 데이터프레임 변환 및 시각화
if len(rms_list) > 0:
    rms_df = pd.DataFrame(rms_list, columns=['B1', 'B2', 'B3', 'B4'])
    rms_df = rms_df.replace([np.inf, -np.inf], np.nan).dropna()  # 무한대/결측치 제거

    print(f"분석 완료! {len(rms_df)}개의 유효한 데이터 포인트를 찾았습니다.")

    # 그래프 그리기
    plt.figure(figsize=(12, 6))

    # IMS Test 1에서 고장이 발생하는 베어링은 주로 B3(내륜)와 B4(전동체).
    plt.plot(rms_df['B3'], label='Bearing 3 (Inner Race)', color='230blue')
    plt.plot(rms_df['B4'], label='Bearing 4 (Roller)', color='green')
    # plt.plot(rms_df['B3'], label='Bearing 3 (Inner Race Fault)')  # 예시: B3 고장
    # plt.axhline(y=rms_df['B3'].iloc[:100].mean() * 1.7, color='r', linestyle='--', label='IQR 170% Threshold')

    # 임계치 계산 (초반 20개 샘플 평균 기준)
    base_mean = rms_df['B3'].iloc[:20].mean()
    plt.axhline(y=base_mean * 1.7, color='red', linestyle='--', label='170% Threshold')

    plt.title('Bearing Vibration Trend (RMS)')
    plt.xlabel('File Index (Time)')
    plt.ylabel('RMS Amplitude')
    plt.legend()
    # plt.grid(True)
    plt.grid(True, alpha=0.3)

    # 중요: 그래프 창이 강제로 맨 앞으로 오게 함
    plt.tight_layout()
    print("그래프를 출력합니다. 창이 뜰 때까지 기다려주세요...")
    plt.show()  # 이 코드에서 창이 뜰 때까지 프로그램이 대기합니다.
else:
    print("분석된 데이터가 없습니다.")