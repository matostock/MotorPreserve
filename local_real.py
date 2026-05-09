import os
import time
import socket
import json
import pandas as pd
import numpy as np

# 설정
IP = '127.0.0.1'
PORT = 9000
DATA_DIR = r'C:\Users\user\Documents\AI_Class\Motor_Preserve\test_sensor_to_server'

# 데이터셋별 샘플링 주파수 정의
DATASET_CONFIG = {
    'ISM': 20000,
    'CWRU': 12000
}

# 순차 전송 시나리오 정의 (클래스 순서)
SCENARIO = ['normal', 'ball', 'inner_race', 'outer_race']


def send_data():
    for dataset in ['ISM', 'CWRU']:
        fs = DATASET_CONFIG.get(dataset, 12000)

        for target_class in SCENARIO:
            class_path = os.path.join(DATA_DIR, dataset, target_class)
            if not os.path.exists(class_path):
                continue

            files = sorted(os.listdir(class_path))
            print(f"\n[데이터셋: {dataset}] {target_class} 상태 전송 시작")

            for fname in files:
                file_path = os.path.join(class_path, fname)

                # 데이터 로드 (확장자에 따른 분기)
                if fname.endswith('.csv') or fname.endswith('.txt'):
                    df = pd.read_csv(file_path, sep='\\s+', header=None)
                    # IMS/ISM의 경우 보통 1번 베어링 데이터 사용
                    bearing_data = df.iloc[:, 1].values.astype(np.float64)
                elif fname.endswith('.mat'):
                    from scipy.io import loadmat
                    mat = loadmat(file_path)
                    # CWRU의 경우 키값 탐색 (예: DE_time 포함된 데이터 추출)
                    keys = [k for k in mat.keys() if 'DE_time' in k]
                    if not keys: continue
                    bearing_data = mat[keys[0]].flatten().astype(np.float64)
                else:
                    continue

                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.connect((IP, PORT))

                        # 확장 헤더: 타입:파일명:번호:정답:출처:주파수
                        # 여기서 b_no는 편의상 1로 고정
                        header = f"RAW_S:{fname}:1:{target_class}:{dataset}:{fs}\n".encode()
                        payload = header + bearing_data.tobytes()
                        s.sendall(payload)
                        print(f"  > 전송 완료: {fname} (Label: {target_class})")

                except Exception as e:
                    print(f"  > Error: {e}")

                time.sleep(0.5)  # 시각화를 위한 간격 조정

send_data()