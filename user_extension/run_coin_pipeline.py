# 전체 파이프라인 실행 예제
from user_extension.binance_data import load_binance_data
from user_extension.lstm_model import predict_price
from user_extension.coin_strategy import execute_trade

def main():
    df = load_binance_data("BTC/USDT", timeframe="1m", limit=200)
    prediction = predict_price(df)
    execute_trade(prediction)

if __name__ == "__main__":
    main()
