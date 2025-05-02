
import torch
from lstm_dualbranch_model import DualBranchLSTM

def save_model(model, path="saved_model.pt"):
    torch.save(model.state_dict(), path)
    print(f"✅ 모델 저장 완료: {path}")

def load_model(input_size, hidden_size=128, path="saved_model.pt", device="cpu"):
    model = LSTMMultiTaskModel(input_size=input_size, hidden_size=hidden_size)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    print(f"✅ 모델 로드 완료: {path}")
    return model

def predict(model, X_input):
    with torch.no_grad():
        X_tensor = torch.tensor(X_input, dtype=torch.float32).unsqueeze(0).to(next(model.parameters()).device)
        entry_prob, direction_prob = model(X_tensor)
        entry = int(entry_prob.item() > 0.5)
        direction = int(direction_prob.argmax(dim=1).item())
        return entry, direction
