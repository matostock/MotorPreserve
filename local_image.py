import os
import time
import socket
import random

# 설정
IP = '127.0.0.1'
PORT = 9000
# 이미지가 저장된 루트 경로
BASE_DATA_DIR = './spectrogram_for_learning/test'
# 전송할 클래스 순서
SCENARIO = ['normal', 'ball', 'inner_race', 'outer_race']


def send_images():
    all_files = []
    for target_class in SCENARIO:
        class_path = os.path.join(BASE_DATA_DIR, target_class)

        if not os.path.exists(class_path):
            print(f"[경고] 경로 없음: {class_path}")
            continue

        # files = sorted([f for f in os.listdir(class_path) if f.endswith('.png')])
        files = [f for f in os.listdir(class_path) if f.endswith('.png')]
        # print(f"\n[{target_class}] 이미지 전송 시작 (총 {len(files)}개)")

        # for fname in files:
        #     file_path = os.path.join(class_path, fname)

        for fname in files:
            # (파일명, 정답 라벨) 튜플 형태로 저장
            all_files.append((fname, target_class))

    # 2. 수집된 리스트의 순서를 무작위로 섞음 (핵심 로직)
    print(f"총 {len(all_files)}개의 파일을 무작위로 섞습니다...")
    random.shuffle(all_files)

    # 3. 섞인 리스트 순서대로 전송
    for fname, target_class in all_files:
        class_path = os.path.join(BASE_DATA_DIR, target_class)
        file_path = os.path.join(class_path, fname)

        try:
            # 1. 이미지 파일을 바이너리(bytes)로 읽기
            with open(file_path, 'rb') as f:
                image_bytes = f.read()

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(5)
                s.connect((IP, PORT))

                # 2. 헤더 구성 (IMG_S:파일명:정답)
                # 서버에서 데이터 타입을 구분할 수 있도록 'IMG_S' 태그 사용
                header = f"IMG_S:{fname}:{target_class}\n".encode()
                payload = header + image_bytes

                s.sendall(payload)
                print(f"  > 전송 완료: {fname} (Label: {target_class})")

        except Exception as e:
            print(f"  > 전송 오류 ({fname}): {e}")

        # 전송 간격 조절 (서버 처리 속도 및 시각화 확인용)
        time.sleep(0.1)


send_images()