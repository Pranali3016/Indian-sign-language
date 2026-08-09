"""
train_robust_classifier.py
Trains the Universal Invariant Sign Language Classifier (SignNet).

Architecture:
 - Input: 162 invariant features (Primary Hand Shape, Elevation, Secondary Hand, Body Posture)
 - 2-layer Bidirectional GRU (Hidden Dim: 64) with LayerNorm and Dropout (0.25)
 - Residual Multi-Head Classification Layer
 - Data Augmentation: Horizontal mirroring, speed jittering, temporal shifts
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from features import extract_from_225_vector, FEATURE_DIM

DATA_PATH = "Data_Mg"
ACTIONS = np.array(['Alone', 'Call', 'Flower', 'Food', 'I am good', 'Ok Fine', 'Stop', 'There is Gun'])
SEQUENCE_LENGTH = 30

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
        out = out[:, -1, :]  # Final temporal state
        return self.fc(out)

def load_dataset():
    sequences = []
    labels = []
    label_map = {action: idx for idx, action in enumerate(ACTIONS)}

    print(f"Loading and processing dataset from '{DATA_PATH}' into Invariant 162-D space...")
    for action in ACTIONS:
        action_dir = os.path.join(DATA_PATH, action)
        if not os.path.exists(action_dir):
            continue

        seq_folders = sorted(os.listdir(action_dir), key=lambda x: int(x) if x.isdigit() else x)
        action_count = 0

        for seq_folder in seq_folders:
            seq_path = os.path.join(action_dir, seq_folder)
            if not os.path.isdir(seq_path):
                continue

            raw_window = []
            for frame_num in range(SEQUENCE_LENGTH):
                frame_file = os.path.join(seq_path, f"{frame_num}.npy")
                if os.path.exists(frame_file):
                    raw_window.append(np.load(frame_file).astype(np.float32))
                else:
                    raw_window.append(np.zeros(225, dtype=np.float32))

            if len(raw_window) == SEQUENCE_LENGTH:
                # 1. Convert to invariant 162-D features
                invariant_seq = [extract_from_225_vector(frame) for frame in raw_window]
                invariant_seq = np.array(invariant_seq, dtype=np.float32)

                # Filter active frames (resample if first half was zero)
                hand_energy = np.sum(np.abs(invariant_seq[:, :63]), axis=1)
                active_idx = np.where(hand_energy > 0.1)[0]
                if len(active_idx) >= 5:
                    start_i = max(0, active_idx[0] - 2)
                    end_i = min(len(invariant_seq), active_idx[-1] + 3)
                    sliced = invariant_seq[start_i:end_i]
                    # Interpolate to 30 frames
                    idx_resample = np.linspace(0, len(sliced) - 1, SEQUENCE_LENGTH)
                    resampled = []
                    for idx_val in idx_resample:
                        l = int(np.floor(idx_val))
                        h = int(np.ceil(idx_val))
                        w = idx_val - l
                        resampled.append((1.0 - w) * sliced[l] + w * sliced[h])
                    invariant_seq = np.array(resampled, dtype=np.float32)

                # Add Original Sequence
                sequences.append(invariant_seq)
                labels.append(label_map[action])

                # Add Horizontally Mirrored Augmentation
                mirrored_seq = invariant_seq.copy()
                # Flip x coordinates in hand shape (first 63 are x,y,z of 21 joints)
                for t in range(SEQUENCE_LENGTH):
                    mirrored_seq[t, 0:63:3] = -mirrored_seq[t, 0:63:3] # Primary hand x
                    mirrored_seq[t, 63] = -mirrored_seq[t, 63]         # Primary pos x
                    mirrored_seq[t, 66:129:3] = -mirrored_seq[t, 66:129:3] # Secondary hand x
                    mirrored_seq[t, 129] = -mirrored_seq[t, 129]       # Secondary pos x
                sequences.append(mirrored_seq)
                labels.append(label_map[action])

                # Add Temporal Jitter (+2 frames shift)
                shifted_seq = np.roll(invariant_seq, shift=2, axis=0)
                shifted_seq[:2] = shifted_seq[2]
                sequences.append(shifted_seq)
                labels.append(label_map[action])

                action_count += 1

        print(f"  [+] Action '{action}': {action_count} raw -> {action_count*3} invariant samples.")

    X = np.array(sequences, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    print(f"\n[INFO] Total Dataset Shape: X = {X.shape}, y = {y.shape} (Feature Dim: {FEATURE_DIM})")
    return X, y

def main():
    X, y = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SignNet(input_dim=FEATURE_DIM, hidden_dim=64, num_classes=len(ACTIONS)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=70, eta_min=1e-5)

    best_acc = 0.0
    print("\nTraining SignNet Invariant Gesture Model...")
    for epoch in range(1, 71):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(batch_y)
            train_correct += (outputs.argmax(dim=1) == batch_y).sum().item()
            train_total += len(batch_y)

        scheduler.step()

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                preds = model(batch_x).argmax(dim=1)
                val_correct += (preds == batch_y).sum().item()
                val_total += len(batch_y)

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total

        if epoch % 10 == 0 or epoch == 1 or epoch == 70:
            print(f"Epoch [{epoch:02d}/70] - Loss: {train_loss/train_total:.4f} | Train Acc: {train_acc*100:.1f}% | Val Acc: {val_acc*100:.1f}%")

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_sign_model.pt")

    print(f"\n[DONE] Best Invariant Model Validation Accuracy: {best_acc*100:.2f}%")
    print("Saved weights to 'best_sign_model.pt'")

    os.makedirs("static", exist_ok=True)
    metadata = {
        "actions": ACTIONS.tolist(),
        "sequence_length": SEQUENCE_LENGTH,
        "input_dim": FEATURE_DIM,
        "hidden_dim": 64,
        "num_classes": len(ACTIONS)
    }
    with open(os.path.join("static", "model_metadata.json"), "w") as f:
        json.dump(metadata, f)
    print("Exported metadata to 'static/model_metadata.json'")

if __name__ == "__main__":
    main()
