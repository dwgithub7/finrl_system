# Binance 데이터 수집 예제 (ccxt 사용)
import ccxt
import pandas as pd

def load_binance_data(symbol="BTC/USDT", timeframe='1m', limit=100):
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df
