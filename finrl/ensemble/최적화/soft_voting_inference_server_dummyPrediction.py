from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import uvicorn

# ✅ FastAPI 기반 Dummy Soft Voting Inference 서버 (Confidence Margin 적용)

class PredictRequest(BaseModel):
    minute_features: list
    daily_features: list

app = FastAPI()

# ✅ Dummy 모델 예측 함수 (랜덤 확률 반환)
def dummy_predict(features, model_name):
    return np.random.uniform(0.4, 0.7)  # 0.4 ~ 0.7 사이 랜덤 확률값

# ✅ 모델 이름 및 가중치 설정
weights = np.array([0.4, 0.3, 0.3])
model_names = ['lstm', 'lightgbm', 'tabnet']

# ✅ Threshold 및 Confidence Margin 설정
THRESHOLD_ENTRY = 0.5
THRESHOLD_DIRECTION = 0.5
CONFIDENCE_MARGIN = 0.05

@app.post("/predict")
def predict(req: PredictRequest):
    try:
        minute_input = np.array(req.minute_features)
        daily_input = np.array(req.daily_features)

        entry_preds = []
        direction_preds = []

        for model_name in model_names:
            entry_prob = dummy_predict(minute_input, model_name)
            direction_prob = dummy_predict(daily_input, model_name)
            entry_preds.append(entry_prob)
            direction_preds.append(direction_prob)

        # ✅ Soft Voting Weighted Average
        final_entry_prob = np.dot(entry_preds, weights)
        final_direction_prob = np.dot(direction_preds, weights)

        # ✅ Confidence Margin 기반 Entry/Direction 결정
        if final_entry_prob > (THRESHOLD_ENTRY + CONFIDENCE_MARGIN):
            entry_signal = 1
            direction_signal = "long" if final_direction_prob > (THRESHOLD_DIRECTION + CONFIDENCE_MARGIN) else "short"
        else:
            entry_signal = 0
            direction_signal = "none"

        # ✅ 슬리피지 및 수수료 반영 예상 수익률 계산 (샘플)
        slippage_rate = 0.0005
        fee_rate = 0.0005

        if direction_signal == "long":
            expected_profit = (1.002 - 1.0) - (2 * fee_rate)
        elif direction_signal == "short":
            expected_profit = (1.0 - 0.998) - (2 * fee_rate)
        else:
            expected_profit = 0.0

        return {
            "entry": entry_signal,
            "direction": direction_signal,
            "expected_profit": round(expected_profit, 5)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
