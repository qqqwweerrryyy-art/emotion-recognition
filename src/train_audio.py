import argparse
import json
import random
import shutil
import subprocess
from pathlib import Path

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import Wav2Vec2Model

CLASSES = ["neutral", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
CODE2CLS = {1: 0, 2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6}  # calm -> neutral
SR = 16000
MAX_LEN = 4 * SR


def split_actors(actors, seed):  # та же логика, что в train_face.py
    actors = sorted(actors)
    random.Random(seed).shuffle(actors)
    n = len(actors)
    n_test = round(0.2 * n) if n >= 6 else 0
    n_val = max(1, round(0.2 * n))
    return actors[n_test + n_val:], actors[n_test:n_test + n_val], actors[:n_test]


def extract_wav(video, out):
    if out.exists():
        return
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video),
           "-vn", "-ac", "1", "-ar", str(SR), str(out)]
    subprocess.run(cmd, check=True)


def load_wave(path):
    y, _ = librosa.load(path, sr=SR)
    y, _ = librosa.effects.trim(y, top_db=30)
    if len(y) < 2048:
        y = np.pad(y, (0, 2048 - len(y)))
    y = (y - y.mean()) / (y.std() + 1e-7)
    return y.astype(np.float32)


class AudioDS(Dataset):
    def __init__(self, waves, labels, train):
        self.waves, self.labels, self.train = waves, labels, train

    def __len__(self):
        return len(self.waves)

    def __getitem__(self, i):
        y = self.waves[i]
        if len(y) > MAX_LEN:
            start = random.randint(0, len(y) - MAX_LEN) if self.train else (len(y) - MAX_LEN) // 2
            y = y[start:start + MAX_LEN]
        elif len(y) < MAX_LEN:
            y = np.pad(y, (0, MAX_LEN - len(y)), mode="wrap")
        return torch.from_numpy(y), self.labels[i]


class Net(nn.Module):
    def __init__(self, n_classes, n_unfreeze):
        super().__init__()
        self.backbone = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.n_unfreeze = n_unfreeze
        if n_unfreeze > 0:
            for layer in self.backbone.encoder.layers[-n_unfreeze:]:
                for p in layer.parameters():
                    p.requires_grad = True
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(768, n_classes))

    def set_train(self):
        self.eval()
        self.head.train()
        if self.n_unfreeze > 0:
            for layer in self.backbone.encoder.layers[-self.n_unfreeze:]:
                layer.train()

    def forward(self, x):
        h = self.backbone(x).last_hidden_state
        return self.head(h.mean(1))


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, probs = [], []
    for x, y in loader:
        x = x.to(device)
        with torch.autocast(device_type=device, enabled=(device == "cuda")):
            out = model(x)
        probs.append(torch.softmax(out.float(), 1).cpu().numpy())
        ys.extend(y.tolist())
    return np.array(ys), np.concatenate(probs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/ravdess")
    ap.add_argument("--audio", default="data/audio")
    ap.add_argument("--mode", choices=["frozen", "partial"], default="frozen")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--lr_layers", type=float, default=5e-5)
    ap.add_argument("--unfreeze", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg не найден в PATH")

    videos = sorted(Path(args.data).rglob("01-*.mp4"))
    print(f"Найдено видео: {len(videos)}")
    audio_dir = Path(args.audio)
    audio_dir.mkdir(parents=True, exist_ok=True)

    stems, waves, labels, actors = [], [], [], []
    for v in tqdm(videos, desc="загрузка звука"):
        wav = audio_dir / f"{v.stem}.wav"
        extract_wav(v, wav)
        parts = v.stem.split("-")
        stems.append(v.stem)
        waves.append(load_wave(wav))
        labels.append(CODE2CLS[int(parts[2])])
        actors.append(int(parts[6]))
    labels, actors = np.array(labels), np.array(actors)

    tr_a, va_a, te_a = split_actors(sorted(set(actors.tolist())), args.seed)
    print(f"Актёры: train={tr_a} val={va_a} test={te_a}")

    def make(actor_list, train):
        idx = [i for i in range(len(waves)) if actors[i] in actor_list]
        ds = AudioDS([waves[i] for i in idx], [int(labels[i]) for i in idx], train)
        dl = DataLoader(ds, batch_size=args.batch, shuffle=train, num_workers=0)
        return dl, idx

    dl_tr, _ = make(tr_a, True)
    dl_va, idx_va = make(va_a, False)
    dl_te, idx_te = make(te_a, False) if te_a else (None, [])

    n_unfreeze = args.unfreeze if args.mode == "partial" else 0
    model = Net(len(CLASSES), n_unfreeze).to(device)
    groups = [{"params": model.head.parameters(), "lr": args.lr_head}]
    if n_unfreeze > 0:
        layer_params = [p for l in model.backbone.encoder.layers[-n_unfreeze:] for p in l.parameters()]
        groups.append({"params": layer_params, "lr": args.lr_layers})
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler(device, enabled=(device == "cuda"))
    crit = nn.CrossEntropyLoss()

    Path("outputs").mkdir(exist_ok=True)
    ckpt = Path(f"outputs/audio_w2v_{args.mode}.pth")
    best_f1 = -1.0

    for ep in range(1, args.epochs + 1):
        model.set_train()
        total, n = 0.0, 0
        for x, y in tqdm(dl_tr, desc=f"epoch {ep}/{args.epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.autocast(device_type=device, enabled=(device == "cuda")):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item() * len(y)
            n += len(y)
        sched.step()
        yv, pv = predict(model, dl_va, device)
        acc, f1 = accuracy_score(yv, pv.argmax(1)), f1_score(yv, pv.argmax(1), average="macro")
        print(f"epoch {ep:2d}: train_loss={total / n:.3f}  val_acc={acc:.3f}  val_f1={f1:.3f}")
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), ckpt)

    model.load_state_dict(torch.load(ckpt, map_location=device))
    yv, pv = predict(model, dl_va, device)
    rows = [(stems[i], int(actors[i]), "val", int(labels[i]), *pv[k]) for k, i in enumerate(idx_va)]
    if dl_te:
        y, p = predict(model, dl_te, device)
        rows += [(stems[i], int(actors[i]), "test", int(labels[i]), *p[k]) for k, i in enumerate(idx_te)]
        name = "test"
    else:
        y, p, name = yv, pv, "val"
    pd.DataFrame(rows, columns=["video", "actor", "split", "label"] + [f"p_{c}" for c in CLASSES]) \
        .to_csv(f"outputs/audio_{args.mode}_probs.csv", index=False)

    pred = p.argmax(1)
    acc, f1 = accuracy_score(y, pred), f1_score(y, pred, average="macro")
    print(f"\nИТОГ ({name}, wav2vec2 {args.mode}): accuracy={acc:.3f}  macro-F1={f1:.3f}")
    print(classification_report(y, pred, labels=range(len(CLASSES)),
                                target_names=CLASSES, zero_division=0))

    cm = confusion_matrix(y, pred, labels=range(len(CLASSES)))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_xlabel("предсказано")
    ax.set_ylabel("истина")
    fig.tight_layout()
    fig.savefig(f"outputs/audio_w2v_{args.mode}_cm.png", dpi=150)
    with open(f"outputs/audio_w2v_{args.mode}_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"mode": args.mode, "split": name, "accuracy": acc, "macro_f1": f1,
                   "train_actors": tr_a, "val_actors": va_a, "test_actors": te_a},
                  f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
