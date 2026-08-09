import os
import json
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from features import extract_from_225_vector, FEATURE_DIM

app = FastAPI(title="SignBridge AI - Universal Invariant Sign Language Recognition")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ACTIONS = ['Alone', 'Call', 'Flower', 'Food', 'I am good', 'Ok Fine', 'Stop', 'There is Gun']
ACTION_INFO = {
    'Alone': {"emoji": "🧍", "meaning": "Feeling or being by oneself", "hint": "Point index finger up or crossed hand sign"},
    'Call': {"emoji": "📞", "meaning": "Calling on phone / reaching out", "hint": "Thumb & pinky extended near the ear"},
    'Flower': {"emoji": "🌸", "meaning": "Flower blossom sign", "hint": "Fingers together opening upward like a blooming flower"},
    'Food': {"emoji": "🍲", "meaning": "Food / eating", "hint": "Hand moving near mouth as if eating"},
    'I am good': {"emoji": "👍", "meaning": "Feeling good / well", "hint": "Thumbs up affirmation near chest"},
    'Ok Fine': {"emoji": "👌", "meaning": "Agreement / everything is fine", "hint": "Index finger and thumb touching (OK sign)"},
    'Stop': {"emoji": "✋", "meaning": "Stop / halt", "hint": "Open palm extended forward or downward chop"},
    'There is Gun': {"emoji": "⚠️", "meaning": "Emergency / weapon alert", "hint": "Hand shaped like a pointer / alert sign"}
}

class SignNet(nn.Module):
    def __init__(self, input_dim=162, hidden_dim=64, num_classes=8):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return self.fc(out)

model = None
MODEL_PATH = "best_sign_model.pt"

def get_model():
    global model
    if model is None and os.path.exists(MODEL_PATH):
        try:
            m = SignNet(input_dim=FEATURE_DIM, hidden_dim=64, num_classes=len(ACTIONS))
            m.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
            m.eval()
            model = m
            print(f"[INFO] SignNet Invariant Model loaded successfully (Input Dim: {FEATURE_DIM}).")
        except Exception as e:
            print(f"[WARN] Error loading PyTorch model: {e}")
    return model

os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

class SequenceInput(BaseModel):
    sequence: List[List[float]]

@app.get("/", response_class=HTMLResponse)
async def serve_home():
    template_path = os.path.join("templates", "index.html")
    if os.path.exists(template_path):
        with open(template_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Sign Language Web App Loading...</h1>")

@app.get("/api/actions")
async def get_actions():
    return {
        "actions": ACTIONS,
        "details": ACTION_INFO,
        "sequence_length": 30,
        "features": FEATURE_DIM
    }

@app.post("/api/predict")
async def predict_sequence(data: SequenceInput):
    m = get_model()
    if m is None:
        return JSONResponse(status_code=503, content={"error": "Model not ready yet."})

    # Convert incoming 225-dim frames into 162-dim Invariant Hand & Elevation space
    invariant_sequence = [extract_from_225_vector(frame) for frame in data.sequence]
    seq_arr = np.array(invariant_sequence, dtype=np.float32)

    with torch.no_grad():
        input_tensor = torch.tensor(seq_arr).unsqueeze(0)
        logits = m(input_tensor)
        probabilities = torch.softmax(logits, dim=1).numpy()[0]

    pred_idx = int(np.argmax(probabilities))
    confidence = float(probabilities[pred_idx])
    probs_dict = {action: float(prob) for action, prob in zip(ACTIONS, probabilities)}

    return {
        "predicted_action": ACTIONS[pred_idx],
        "confidence": confidence,
        "probabilities": probs_dict
    }

if __name__ == "__main__":
    import uvicorn
    print("Starting Sign Language Live AI Server on http://localhost:8050 ...")
    uvicorn.run("app:app", host="0.0.0.0", port=8050, reload=False)
