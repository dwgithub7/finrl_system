# 판단 로직 + 주문 실행 구조 예시
def execute_trade(prediction):
    if prediction == 1:
        print("🚀 매수 시그널 발생! 실제 거래 코드로 연결하세요.")
    elif prediction == -1:
        print("📉 매도 시그널 발생!")
    else:
        print("⏸ 관망 상태")
