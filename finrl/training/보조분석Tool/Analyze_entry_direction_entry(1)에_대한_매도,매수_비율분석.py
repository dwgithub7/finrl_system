import numpy as np
import os

# ✅ 설정값
DATA_DIR = "data/dualbranch"
SYMBOL_PERIOD = "SOLUSDT_1m_finrl_20240101~20250426"

# ✅ 데이터 로딩
y_entry = np.load(os.path.join(DATA_DIR, f"{SYMBOL_PERIOD}_y_entry.npy"))
y_direction = np.load(os.path.join(DATA_DIR, f"{SYMBOL_PERIOD}_y_direction.npy"))

# ✅ entry=1 샘플만 추출
mask = y_entry == 1
y_direction_entry = y_direction[mask]

# ✅ 매수(1) / 매도(0) 비율 계산
total = len(y_direction_entry)
count_0 = np.sum(y_direction_entry == 0)
count_1 = np.sum(y_direction_entry == 1)

ratio_0 = count_0 / total
ratio_1 = count_1 / total

print("\n🎯 Entry=1 샘플 중 방향(label) 분포:")
print(f"- 매도(0) 수: {count_0}개 ({ratio_0*100:.2f}%)")
print(f"- 매수(1) 수: {count_1}개 ({ratio_1*100:.2f}%)")
print(f"- 총 Entry=1 샘플 수: {total}개")

# ✅ 경고 표시 (심각한 불균형 여부)
if ratio_0 < 0.3 or ratio_1 < 0.3:
    print("\n🚨 경고: 방향(label) 간 심각한 imbalance(편향) 존재!")
else:
    print("\n✅ 방향(label) 분포는 비교적 균형적입니다.")
