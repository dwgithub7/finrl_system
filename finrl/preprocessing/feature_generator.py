
import numpy as np
import os

def load_dualbranch_data(base_dir, suffix="20250301~20250426"):
    X_minute = np.load(os.path.join(base_dir, f"SOLUSDT_1m_finrl_{suffix}_X_minute_full.npy"))
    X_daily = np.load(os.path.join(base_dir, f"SOLUSDT_1m_finrl_{suffix}_X_daily_full.npy"))
    y_entry = np.load(os.path.join(base_dir, f"SOLUSDT_1m_finrl_{suffix}_y_entry_full.npy"))
    y_direction = np.load(os.path.join(base_dir, f"SOLUSDT_1m_finrl_{suffix}_y_direction_full.npy"))
    return X_minute, X_daily, y_entry, y_direction

def generate_combined_features(X_minute, X_daily):
    X_minute_last = X_minute[:, -1, :]
    X_daily_last = X_daily[:, -1, :]

    price = X_minute_last[:, 0]
    ma20 = X_minute_last[:, 9]
    ma5 = X_minute_last[:, 7]
    rsi = X_minute_last[:, 5]
    macd = X_minute_last[:, 4]

    price_ratio_ma20 = np.expand_dims(price / (ma20 + 1e-6), axis=1)
    ma_diff = np.expand_dims(ma5 - ma20, axis=1)
    rsi_slope = np.expand_dims(rsi, axis=1)
    macd_slope = np.expand_dims(macd, axis=1)

    X_minute_v2 = np.concatenate([X_minute_last[:, :16], price_ratio_ma20, ma_diff, rsi_slope, macd_slope], axis=1)

    price_d = X_daily_last[:, 0]
    ma20_d = X_daily_last[:, 9]
    ma5_d = X_daily_last[:, 7]
    rsi_d = X_daily_last[:, 5]
    macd_d = X_daily_last[:, 4]

    price_ratio_ma20_d = np.expand_dims(price_d / (ma20_d + 1e-6), axis=1)
    ma_diff_d = np.expand_dims(ma5_d - ma20_d, axis=1)
    rsi_slope_d = np.expand_dims(rsi_d, axis=1)
    macd_slope_d = np.expand_dims(macd_d, axis=1)

    X_daily_v2 = np.concatenate([X_daily_last[:, :16], price_ratio_ma20_d, ma_diff_d, rsi_slope_d, macd_slope_d], axis=1)

    return np.concatenate([X_minute_v2, X_daily_v2], axis=1)

def filter_by_entry(X, y_entry, y_direction):
    mask = y_entry == 1
    return X[mask], y_entry[mask], y_direction[mask]
