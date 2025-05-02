import ccxt
import pandas as pd
import os

# 🔥 설정
symbol = "SOL/USDT"
timeframe = "1m"
start_date = "2024-01-01 00:00:00"  # 시작일
end_date = "2025-04-26 23:59:00"    # 종료일

save_path = "data/raw"
os.makedirs(save_path, exist_ok=True)

# 📦 바이낸스 객체 생성
binance = ccxt.binance()
binance.load_markets()

since = binance.parse8601(start_date + 'Z')
end_timestamp = binance.parse8601(end_date + 'Z')
limit = 1000

# 🔵 심볼 포맷 변환: SOL/USDT → SOLUSDT
symbol_name = symbol.replace("/", "")

# 🔵 파일명 포맷 생성 (날짜만 추출)
start_label = start_date[:10].replace("-", "")  # "2024-01-01" -> "20240101"
end_label = end_date[:10].replace("-", "")      # "2025-04-26" -> "20250426"
filename = f"{symbol_name}_{timeframe}_raw_{start_label}~{end_label}.csv"

all_data = []
while since < end_timestamp:
    ohlcv = binance.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
    if not ohlcv:
        break
    for entry in ohlcv:
        if entry[0] > end_timestamp:
            break
        all_data.append(entry)
    since = ohlcv[-1][0] + 60000  # 다음 1분

    if len(all_data) > 1_000_000:
        print("⚠️ 너무 많은 데이터, 조기 종료")
        break

# ✅ DataFrame 변환
df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
df.rename(columns={"timestamp": "date"}, inplace=True)

save_file = os.path.join(save_path, filename)
df.to_csv(save_file, index=False)

print(f"✅ 저장 완료: {save_file} (총 {len(df)}개 행)")

