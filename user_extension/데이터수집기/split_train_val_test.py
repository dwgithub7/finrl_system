
import numpy as np
import os
from sklearn.model_selection import train_test_split

DATA_DIR = "data/multitask/"
SEED = 42
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# 🔹 원본 데이터 로드
X_min = np.load(os.path.join(DATA_DIR, "X_minute.npy"))
X_day = np.load(os.path.join(DATA_DIR, "X_daily.npy"))
y_entry = np.load(os.path.join(DATA_DIR, "y_entry.npy"))
y_dir = np.load(os.path.join(DATA_DIR, "y_direction.npy"))

# 🔹 인덱스 분리
N = len(y_entry)
idx = np.arange(N)

# 우선 test set 분리
idx_train_val, idx_test = train_test_split(idx, test_size=TEST_RATIO, shuffle=False)

# train/val 분리
val_size = int(len(idx_train_val) * VAL_RATIO / (1 - TEST_RATIO))
idx_train, idx_val = idx_train_val[:-val_size], idx_train_val[-val_size:]

def save_split(name, idxs):
    np.save(DATA_DIR + f"{name}_X_minute.npy", X_min[idxs])
    np.save(DATA_DIR + f"{name}_X_daily.npy", X_day[idxs])
    np.save(DATA_DIR + f"{name}_y_entry.npy", y_entry[idxs])
    np.save(DATA_DIR + f"{name}_y_direction.npy", y_dir[idxs])
    print(f"✅ {name.upper()} 저장 완료: {len(idxs)}개 샘플")

save_split("train", idx_train)
save_split("val", idx_val)
save_split("test", idx_test)
