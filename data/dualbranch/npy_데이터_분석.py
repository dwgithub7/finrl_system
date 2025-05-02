import numpy as np
x = np.load('data/dualbranch/SOLUSDT_1m_finrl_20250101~20250426_X_minute_full.npy', allow_pickle=True)
x2 = np.load('data/dualbranch/SOLUSDT_1m_finrl_20250101~20250426_X_daily_full.npy', allow_pickle=True)
x3 = np.load('data/dualbranch/SOLUSDT_1m_finrl_20250101~20250426_y_direction_full.npy', allow_pickle=True)
x4 = np.load('data/dualbranch/SOLUSDT_1m_finrl_20250101~20250426_y_entry_full.npy', allow_pickle=True)
print(x.dtype)  # 🔥 여기서 dtype이 'object'로 나온다면 문제
print(x.shape)
print(x2.dtype)  # 🔥 여기서 dtype이 'object'로 나온다면 문제
print(x2.shape)
print(x3.dtype)  # 🔥 여기서 dtype이 'object'로 나온다면 문제
print(x3.shape)
print(x4.dtype)  # 🔥 여기서 dtype이 'object'로 나온다면 문제
print(x4.shape)