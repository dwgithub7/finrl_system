from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
import lightgbm as lgb
from pytorch_tabnet.tab_model import TabNetClassifier
import uvicorn

# ✅ FastAPI 기반 최종 Soft Voting Inference 서버 (Confidence Margin 적용)

class PredictRequest(BaseModel):
    minute_features: list
    daily_features: list

app = FastAPI()

class ModelLoader:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_models()

    def load_models(self):
        self.lstm_model = torch.load("lstm_model.pth", map_location=self.device)
        self.lstm_model.eval()

        self.lgb_model = lgb.Booster(model_file="lightgbm_model.txt")

        self.tabnet_model = TabNetClassifier()
        self.tabnet_model.load_model("tabnet_model.zip")

    def predict_lstm(self, minute_features):
        with torch.no_grad():
            input_tensor = torch.tensor(minute_features, dtype=torch.float32).unsqueeze(0).to(self.device)
            entry_prob, direction_prob = self.lstm_model(input_tensor)
            entry_prob = torch.sigmoid(entry_prob).cpu().numpy().flatten()[0]
            direction_prob = torch.softmax(direction_prob, dim=1).cpu().numpy().flatten()[1]
        return entry_prob, direction_prob

    def predict_lightgbm(self, features):
        pred = self.lgb_model.predict(np.array(features).reshape(1, -1))
        return pred[0], pred[1]

    def predict_tabnet(self, features):
        preds = self.tabnet_model.predict_proba(np.array(features).reshape(1, -1))
        entry_prob = preds[0, 1]
        direction_prob = preds[0, 1]
        return entry_prob, direction_prob

model_loader = ModelLoader()

# ✅ 모델별 Soft Voting 가중치 설정
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

        entry_lstm, direction_lstm = model_loader.predict_lstm(minute_input)
        entry_preds.append(entry_lstm)
        direction_preds.append(direction_lstm)

        entry_lgb, direction_lgb = model_loader.predict_lightgbm(np.concatenate([minute_input, daily_input]))
        entry_preds.append(entry_lgb)
        direction_preds.append(direction_lgb)

        entry_tabnet, direction_tabnet = model_loader.predict_tabnet(np.concatenate([minute_input, daily_input]))
        entry_preds.append(entry_tabnet)
        direction_preds.append(direction_tabnet)

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
