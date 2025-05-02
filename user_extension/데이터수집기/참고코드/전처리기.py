import pandas as pd
import numpy as np
import os
import time
import traceback

RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
FILENAME = "SOLUSDT_future_1m_200001010000~202312312359.csv"

def preprocess_debug():
    start = time.time()

    filepath = os.path.join(RAW_DATA_DIR, FILENAME)
    base, ext = os.path.splitext(FILENAME)
    processed_filename = f"{base}_processed.csv.gz"
    processed_path = os.path.join(PROCESSED_DATA_DIR, processed_filename)

    print("📁 입력 파일 경로:", filepath)


    try:
        if not os.path.exists(filepath):
            print(f"❌ 파일이 존재하지 않음: {filepath}")
            return

        # CSV 로딩
        print("📥 CSV 읽는 중...")
        df = pd.read_csv(filepath, engine="pyarrow")
        print(f"✅ CSV 로딩 완료: {df.shape}")

        # 컬럼 설정
        df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)

        # 결측/이상 제거
        df.dropna(inplace=True)
        df = df[(df["volume"] > 0) & (df["close"] > 0)]

        # 피처 생성
        df["return"] = df["close"].pct_change().clip(-0.1, 0.1)
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        df["ma_5"] = df["close"].rolling(5).mean()
        df["ma_20"] = df["close"].rolling(20).mean()
        df["volatility_5"] = df["log_return"].rolling(5).std() * np.sqrt(5)

        # 디렉토리 생성
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

        # 저장
        df.to_csv(processed_path, compression="gzip")
        print(f"💾 저장 완료: {processed_path}")

    except Exception as e:
        print("❌ 예외 발생!")
        print(traceback.format_exc())  # 전체 에러 메시지 출력
        return

    print(f"⏱ 총 소요시간: {time.time() - start:.2f}초")


