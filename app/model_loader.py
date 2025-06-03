import torch
from your_model_file import MultiStreamKeyPredictor  # 🔥 반드시 경로 맞춰 수정

def load_model(checkpoint_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultiStreamKeyPredictor(d_model=128, nhead=4, num_layers=4)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval().to(device)
    return model

