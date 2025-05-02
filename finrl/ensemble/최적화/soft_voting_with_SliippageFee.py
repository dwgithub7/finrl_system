import numpy as np

# ✅ 가정된 슬리피지 및 수수료 설정
slippage_rate = 0.0005  # 슬리피지 0.05%
fee_rate = 0.0005       # 수수료 0.05%

# ✅ 예시: 매수가/매도가 (Soft Voting 예측으로 가정)
np.random.seed(42)
entry_prices = np.random.uniform(100, 200, size=1000)  # 진입 가격 랜덤 생성
exit_prices = entry_prices * np.random.uniform(0.98, 1.02, size=1000)  # 청산 가격 랜덤 생성

# ✅ 예시: 진입 방향 (1=롱, 0=숏)
entry_directions = np.random.randint(0, 2, size=1000)

# ✅ 수익률 계산 함수
def calculate_profit(entry_prices, exit_prices, directions, slippage_rate=0.0005, fee_rate=0.0005):
    adjusted_entry_prices = np.where(
        directions == 1,
        entry_prices * (1 + slippage_rate),  # 롱이면 매수가 + 슬리피지
        entry_prices * (1 - slippage_rate)   # 숏이면 매수가 - 슬리피지
    )

    adjusted_exit_prices = np.where(
        directions == 1,
        exit_prices * (1 - slippage_rate),  # 롱이면 매도가 - 슬리피지
        exit_prices * (1 + slippage_rate)   # 숏이면 매도가 + 슬리피지
    )

    # 수익률 계산
    long_profit = (adjusted_exit_prices - adjusted_entry_prices) / adjusted_entry_prices
    short_profit = (adjusted_entry_prices - adjusted_exit_prices) / adjusted_exit_prices

    raw_profit = np.where(directions == 1, long_profit, short_profit)

    # 수수료 반영 (매수 + 매도 2번 발생)
    final_profit = raw_profit - (2 * fee_rate)

    return final_profit

# ✅ 수익률 계산
profits = calculate_profit(entry_prices, exit_prices, entry_directions)

# ✅ 최종 결과 요약
total_return = np.prod(1 + profits) - 1  # 전체 누적 수익률
average_profit = np.mean(profits)
positive_rate = np.sum(profits > 0) / len(profits)

print("===== 수익률 계산 결과 (슬리피지/수수료 반영) =====")
print(f"전체 누적 수익률: {total_return*100:.2f}%")
print(f"평균 수익률: {average_profit*100:.4f}%")
print(f"수익 거래 비율: {positive_rate*100:.2f}%")
