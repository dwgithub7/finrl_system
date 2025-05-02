# binance_ohlcv_downloader.py
import ccxt
import pandas as pd
import os
from datetime import datetime

# 설정
symbol = "SOL/USDT"
timeframe = "1m"
save_dir = "data/raw"
os.makedirs(save_dir, exist_ok=True)

start_date = "2025-01-01T00:00:00Z"
end_date = "2025-04-26T00:00:00Z"

exchange = ccxt.binance({
    'options': {'defaultType': 'future'}  # 현물 : spot, 선물 : future
})

def fetch_data(symbol, timeframe, since, limit=1000):
    all_data = []
    while since < exchange.parse8601(end_date):
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        if not ohlcv:
            break
        all_data += ohlcv
        since = ohlcv[-1][0] + 60000
    return all_data

since = exchange.parse8601(start_date)
data = fetch_data(symbol, timeframe, since)

df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
df = df[["date", "open", "high", "low", "close", "volume"]]

start_str = datetime.strptime(start_date, "%Y-%m-%dT%H:%M:%SZ").strftime("%Y%m%d")
end_str = datetime.strptime(end_date, "%Y-%m-%dT%H:%M:%SZ").strftime("%Y%m%d")
symbol_formatted = symbol.replace("/", "")

save_path = os.path.join(save_dir, f"{symbol_formatted}_1m_raw_{start_str}~{end_str}.csv")
df.to_csv(save_path, index=False)

print(f"✅ 저장 완료: {save_path} (shape={df.shape})")
