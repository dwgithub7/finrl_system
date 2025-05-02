#MDD, 손절/익절
import numpy as np
import matplotlib.pyplot as plt

# ✅ 샘플 수익률 생성 (Soft Voting 이후 수익률이라고 가정)
np.random.seed(42)
profits = np.random.uniform(-0.05, 0.05, size=1000)  # -5% ~ +5% 랜덤 수익률

# ✅ 손절/익절 기준 설정
cut_loss_threshold = -0.03  # -3% 손절
take_profit_threshold = 0.05  # +5% 익절

# ✅ 손절/익절 적용
def apply_cut_loss_take_profit(profits, cut_loss=-0.03, take_profit=0.05):
    adjusted_profits = np.clip(profits, cut_loss, take_profit)
    return adjusted_profits

adjusted_profits = apply_cut_loss_take_profit(profits, cut_loss_threshold, take_profit_threshold)

# ✅ 누적 수익률 계산 (adjusted)
cumulative_returns = np.cumprod(1 + adjusted_profits)

# ✅ MDD (Maximum Drawdown) 계산
def calculate_mdd(cumulative_returns):
    high_water_mark = np.maximum.accumulate(cumulative_returns)
    drawdowns = (high_water_mark - cumulative_returns) / high_water_mark
    mdd = np.max(drawdowns)
    return mdd, drawdowns

mdd, drawdowns = calculate_mdd(cumulative_returns)

print("===== 리스크 관리 결과 =====")
print(f"최대 낙폭(MDD): {mdd*100:.2f}%")
print(f"최종 누적 수익률: {(cumulative_returns[-1] - 1)*100:.2f}%")

# ✅ 누적 수익률 및 MDD 시각화
plt.figure(figsize=(10,5))
plt.plot(cumulative_returns, label="Cumulative Return")
plt.plot(high_water_mark := np.maximum.accumulate(cumulative_returns), linestyle='--', label="High Water Mark")
plt.fill_between(range(len(cumulative_returns)), cumulative_returns, high_water_mark, color='red', alpha=0.3, label="Drawdown")
plt.title("누적 수익률 및 MDD 시각화")
plt.xlabel("거래 수")
plt.ylabel("수익률 배율")
plt.legend()
plt.grid()
plt.show()
