
import pandas as pd

# ✅ 설정값
INPUT_PATH = "data/processed/sol1m_with_trend.csv"
SHIFT_THRESHOLD = -1  # 미래 정보 포함 여부 판단 기준 (음수면 유출 가능성)

df = pd.read_csv(INPUT_PATH, index_col=0, parse_dates=True)
report = []

for col in df.columns:
    if col.lower().startswith("y") or col.lower().startswith("future"):
        continue
    # shift 방향 추정
    s = df[col].copy()
    autocorr_1 = s.corr(s.shift(1))
    autocorr_m1 = s.corr(s.shift(-1))
    if autocorr_m1 > autocorr_1 + 0.05:
        report.append((col, "❗ 의심: 미래 데이터 유출 가능", round(autocorr_1, 4), round(autocorr_m1, 4)))
    else:
        report.append((col, "✅ 정상", round(autocorr_1, 4), round(autocorr_m1, 4)))

df_report = pd.DataFrame(report, columns=["피처", "유출 여부", "시프트 +1 상관", "시프트 -1 상관"])
df_report = df_report.sort_values("유출 여부", ascending=False)

print("🧪 Feature Leakage 점검 결과:")
print(df_report.to_string(index=False))
