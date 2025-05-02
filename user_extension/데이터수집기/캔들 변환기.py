# resample_candles.py
# 1분봉 CSV를 다양한 N분봉으로 리샘플링하는 도구

import pandas as pd
import os

def resample_ohlcv(input_csv, output_dir="data/resampled/", target_timeframes=["5T", "15T", "1H"], verbose=True):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_csv, parse_dates=["timestamp"])
    df.set_index("timestamp", inplace=True)

    for tf in target_timeframes:
        df_resampled = df.resample(tf).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        }).dropna()

        freq_label = tf.replace("T", "m").replace("H", "h").replace("D", "d")
        filename = os.path.basename(input_csv).replace(".csv", f"_{freq_label}.csv")
        save_path = os.path.join(output_dir, filename)

        df_resampled.to_csv(save_path)
        if verbose:
            print(f"✅ 저장 완료: {save_path} ({len(df_resampled)} rows)")

if __name__ == "__main__":
    # 예시 설정
    input_csv = "data/raw/SOLUSDT_future_1m_200001010000~202312312359.csv"  # 🔧 기존 1분봉 CSV 파일
    output_folder = "data/raw/resampled/"
    target_frames = ["5T", "15T", "1H", "1D"]  # 🔁 생성할 분봉 리스트

    resample_ohlcv(input_csv, output_folder, target_frames)
