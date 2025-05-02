
import pandas as pd
import numpy as np
import os

# ✅ 설정값
INPUT_PATH = "data/processed/sol1m_with_trend.csv"
OUTPUT_DIR = "data/multitask/"

# 🔍 데이터 로딩 및 날짜 확인
df = pd.read_csv(INPUT_PATH, index_col=0, parse_dates=True)

print("✅ 전체 타임스탬프 범위:", df.index.min(), "~", df.index.max())
print("📆 고유 일자 수:", df.index.normalize().nunique())

# 🔍 일봉 집계 및 개수 확인
df["date"] = df.index.date
df_daily = df.groupby("date").mean()
print("📊 생성 가능한 일봉 수:", len(df_daily))
print(df_daily.tail())

# (선택적으로 df.to_csv("디버깅.csv") 저장도 가능)
