import numpy as np

# ✅ Soft Voting 최종 결과 (예시)
final_entry_probs = np.random.uniform(0.4, 0.7, size=100)  # 100개 샘플
final_direction_probs = np.random.uniform(0.4, 0.7, size=100)

# ✅ 최적 Threshold 값 설정
THRESHOLD_ENTRY = 0.5
THRESHOLD_DIRECTION = 0.5
CONFIDENCE_MARGIN = 0.05  # Margin 설정 (예: 5%)

# ✅ Post-processing 함수 정의
def post_process_signals(entry_probs, direction_probs, threshold_entry, threshold_direction, margin):
    entry_signals = []
    direction_signals = []

    for e_prob, d_prob in zip(entry_probs, direction_probs):
        # Entry 결정
        if e_prob > (threshold_entry + margin):
            entry_signal = 1  # 진입
        else:
            entry_signal = 0  # 관망

        # Direction 결정 (entry가 있을 때만 판단)
        if entry_signal == 1:
            direction_signal = "long" if d_prob > (threshold_direction + margin) else "short"
        else:
            direction_signal = "none"

        entry_signals.append(entry_signal)
        direction_signals.append(direction_signal)

    return np.array(entry_signals), np.array(direction_signals)

# ✅ 적용
entry_signals, direction_signals = post_process_signals(
    final_entry_probs,
    final_direction_probs,
    THRESHOLD_ENTRY,
    THRESHOLD_DIRECTION,
    CONFIDENCE_MARGIN
)

# ✅ 결과 확인 (상위 10개 샘플)
print("===== Post-processed 매매 신호 (Top 10) =====")
for i in range(10):
    print(f"[{i}] Entry Prob: {final_entry_probs[i]:.4f}, Direction Prob: {final_direction_probs[i]:.4f}")
    print(f"     Entry Signal: {entry_signals[i]}, Direction Signal: {direction_signals[i]}")
