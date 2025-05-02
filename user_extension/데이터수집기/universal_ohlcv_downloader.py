# universal_ohlcv_downloader.py
import ccxt
import pandas as pd
import os
from datetime import datetime
from time import time, sleep
from pathlib import Path


def fetch_ohlcv(symbol="BTC/USDT", market_type="spot", timeframe='1m', limit=1000, start_time=None, end_time=None, verbose=True):
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {
            'defaultType': market_type  # 'spot' or 'future'
        }
    })

    since = int(pd.Timestamp(start_time).timestamp() * 1000) if start_time else None
    end_ts = pd.Timestamp(end_time).timestamp() * 1000 if end_time else None

    all_data = []
    total_rows = 0
    while True:
        try:
            start_fetch = time()
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit, since=since)
            elapsed = time() - start_fetch

            if not ohlcv:
                if verbose:
                    print("⚠️ 더 이상 데이터를 가져올 수 없습니다.")
                break

            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            all_data.append(df)

            total_rows += len(df)
            if verbose:
                print(f"✅ {symbol} 수집된 행 수: {total_rows} (+{len(df)}) / 최근: {df['timestamp'].iloc[-1]} / 시간: {elapsed:.2f}s")

            last_ts = ohlcv[-1][0]
            if end_ts and last_ts >= end_ts:
                break

            since = last_ts + 60_000
            sleep(0.25)
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
            sleep(5)

    if all_data:
        df_all = pd.concat(all_data).drop_duplicates(subset="timestamp").reset_index(drop=True)
        return df_all
    return pd.DataFrame()

if __name__ == "__main__":
    # 🔧 사용자 설정
    symbol = "SOL/USDT"  # BTC/USDT
    market_type = "future"  # 'spot' 또는 'future'
    timeframe = "1m"
    start = "2000-01-01 00:00:00"
    end = "2023-12-31 23:59:00"
    save_dir = "data/raw/"
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # 수집 실행
    df = fetch_ohlcv(symbol=symbol, market_type=market_type, timeframe=timeframe,
                     start_time=start, end_time=end, verbose=True)

    if not df.empty:
        safe_symbol = symbol.replace("/", "")
        start_str = pd.to_datetime(start).strftime("%Y%m%d%H%M")
        end_str = pd.to_datetime(end).strftime("%Y%m%d%H%M")
        filename = f"{safe_symbol}_{market_type}_{timeframe}_{start_str}~{end_str}.csv"
        file_path = os.path.join(save_dir, filename)

        df.to_csv(file_path, index=False)
        print(f"💾 저장 완료: {file_path} (총 {len(df)}행)")
    else:
        print("❌ 저장 실패 또는 데이터 없음")
