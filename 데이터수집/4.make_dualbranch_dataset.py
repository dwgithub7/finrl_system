import numpy as np
import os
from sklearn.model_selection import train_test_split

# ✅ 설정값
DATA_DIR = "data/dualbranch"
OUTPUT_DIR = "data/testset_dualbranch"
SYMBOL_PERIOD = "SOLUSDT_1m_finrl_20240101~20250426"
TEST_SIZE = 0.2  # 20%를 테스트셋으로 분리

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ✅ 데이터 로딩
X_minute = np.load(os.path.join(DATA_DIR, f"{SYMBOL_PERIOD}_X_minute.npy"))
X_daily = np.load(os.path.join(DATA_DIR, f"{SYMBOL_PERIOD}_X_daily.npy"))
y_entry = np.load(os.path.join(DATA_DIR, f"{SYMBOL_PERIOD}_y_entry.npy"))
y_direction = np.load(os.path.join(DATA_DIR, f"{SYMBOL_PERIOD}_y_direction.npy"))

# ✅ Train/Test 분리
X_minute_train, X_minute_test, X_daily_train, X_daily_test, y_entry_train, y_entry_test, y_direction_train, y_direction_test = train_test_split(
    X_minute, X_daily, y_entry, y_direction, test_size=TEST_SIZE, shuffle=False
)

# ✅ 테스트셋 저장
np.save(os.path.join(OUTPUT_DIR, f"{SYMBOL_PERIOD}_X_minute_test.npy"), X_minute_test)
np.save(os.path.join(OUTPUT_DIR, f"{SYMBOL_PERIOD}_X_daily_test.npy"), X_daily_test)
np.save(os.path.join(OUTPUT_DIR, f"{SYMBOL_PERIOD}_y_entry_test.npy"), y_entry_test)
np.save(os.path.join(OUTPUT_DIR, f"{SYMBOL_PERIOD}_y_direction_test.npy"), y_direction_test)

print(f"✅ 테스트셋 저장 완료: {OUTPUT_DIR}")
print(f"✅ X_minute_test.shape: {X_minute_test.shape}")
print(f"✅ X_daily_test.shape: {X_daily_test.shape}")
print(f"✅ y_entry_test.shape: {y_entry_test.shape}")
print(f"✅ y_direction_test.shape: {y_direction_test.shape}")
