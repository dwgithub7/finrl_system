
# binance_data.py : OHLCV 수집기 (계정 불필요)
import ccxt
import pandas as pd
import time

def load_binance_ohlcv(symbol="BTC/USDT", timeframe="1m", limit=1000):
    exchange = ccxt.binance()
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df
    except Exception as e:
        print(f"❌ 데이터 수집 실패: {e}")
        return pd.DataFrame()

# 사용 예시
if __name__ == "__main__":
    df = load_binance_ohlcv("BTC/USDT", "1m", 1000)
    print(df.tail())
