import argparse
import json
import random
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

CLASSES = ["neutral", "happy", "sad", "angry", "fearful", "disgust"]
CLASS2IDX = {name: i for i, name in enumerate(CLASSES)}
SR = 16000
MAX_LEN = 4 * SR


def load_wave(path):
    y, _ = librosa.load(str(path), sr=SR, mono=True)
    y, _ = librosa.effects.trim(y, top_db=30)
    if len(y) < 2048:
        y = np.pad(y, (0, 2048 - len(y)))
    y = (y - y.mean()) / (y.std() + 1e-7)
    return y.astype(np.float32)


class CremaAudioDS(Dataset):
    def __init__(self, df, train):
        self.paths = df["audio_path"].astype(str).tolist()
        self.labels = [CLASS2IDX[str(x).strip().lower()] for x in df["emotion"]]
        self.train = train

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        y = load_wave(self.paths[i])
        if len(y) > MAX_LEN:
            start = random.randint(0, len(y) - MAX_LEN) if self.train else (len(y) - MAX_LEN) // 2
            y = y[start:start + MAX_LEN]
        elif len(y) < MAX_LEN:
            # Repeat the utterance to a fixed duration for batching.
            repeats = int(np.ceil(MAX_LEN / len(y)))
            y = np.tile(y, repeats)[:MAX_LEN]
        return torch.from_numpy(y.copy()), self.labels[i]


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
        hidden = self.backbone(x).last_hidden_state
        return self.head(hidden.mean(1))


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, probs = [], []
    for x, y in tqdm(loader, desc="оценка", leave=False):
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device, enabled=(device == "cuda")):
            logits = model(x)
        probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        ys.extend(y.tolist())
    return np.asarray(ys), np.concatenate(probs, axis=0)


def main():
    ap = argparse.ArgumentParser(description="Train CREMA-D wav2vec2 emotion model")
    ap.add_argument("--manifest", default="data/cremad_processed/audio_manifest.csv")
    ap.add_argument("--mode", choices=["frozen", "partial"], default="frozen")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--lr_layers", type=float, default=5e-5)
    ap.add_argument("--unfreeze", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit_train", type=int, default=0,
                    help="Для пробного запуска: ограничить число train-записей; 0 = без ограничения")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Устройство: {device}")

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise SystemExit(f"Не найден манифест: {manifest_path}")
    df = pd.read_csv(manifest_path)

    required = {"audio_path", "actor", "emotion", "split"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"В audio_manifest.csv не хватает столбцов: {sorted(missing)}")

    df = df.dropna(subset=["audio_path", "actor", "emotion", "split"]).copy()
    df["emotion"] = df["emotion"].astype(str).str.strip().str.lower()
    df["split"] = df["split"].astype(str).str.strip().str.lower()
    unknown = sorted(set(df["emotion"]) - set(CLASSES))
    if unknown:
        raise SystemExit(f"Обнаружены неизвестные эмоции: {unknown}")
    df = df[df["split"].isin(["train", "val", "test"])].copy()

    # Resolve paths relative to project root (the directory from which the command is run).
    missing_paths = [p for p in df["audio_path"].astype(str) if not Path(p).exists()]
    if missing_paths:
        raise SystemExit(f"Не найдены WAV-файлы, например: {missing_paths[:3]}")

    actor_sets = {s: set(df.loc[df["split"] == s, "actor"].astype(str)) for s in ["train", "val", "test"]}
    if actor_sets["train"] & actor_sets["val"] or actor_sets["train"] & actor_sets["test"] or actor_sets["val"] & actor_sets["test"]:
        raise SystemExit("ОШИБКА: один и тот же актёр попал в несколько split!")
    for split in ["train", "val", "test"]:
        n = int((df["split"] == split).sum())
        print(f"Аудиозаписей {split}: {n}")
        if n == 0:
            raise SystemExit(f"В манифесте нет split={split}")
    print("Актёры по split:", {s: len(v) for s, v in actor_sets.items()})
    print("Классы:", CLASSES)

    train_df = df[df["split"] == "train"].copy()
    if args.limit_train > 0:
        train_df = train_df.sample(n=min(args.limit_train, len(train_df)), random_state=args.seed)
        print(f"ПРОБНЫЙ РЕЖИМ: для обучения взято {len(train_df)} train-записей")

    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    dl_tr = DataLoader(CremaAudioDS(train_df, True), batch_size=args.batch, shuffle=True, num_workers=0)
    dl_va = DataLoader(CremaAudioDS(val_df, False), batch_size=args.batch, shuffle=False, num_workers=0)
    dl_te = DataLoader(CremaAudioDS(test_df, False), batch_size=args.batch, shuffle=False, num_workers=0)

    n_unfreeze = args.unfreeze if args.mode == "partial" else 0
    model = Net(len(CLASSES), n_unfreeze).to(device)
    groups = [{"params": model.head.parameters(), "lr": args.lr_head}]
    if n_unfreeze > 0:
        layer_params = [p for layer in model.backbone.encoder.layers[-n_unfreeze:] for p in layer.parameters()]
        groups.append({"params": layer_params, "lr": args.lr_layers})
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(device, enabled=(device == "cuda"))
    criterion = nn.CrossEntropyLoss()

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    ckpt = out_dir / f"audio_cremad_w2v_{args.mode}.pth"
    best_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        model.set_train()
        total_loss, total_n = 0.0, 0
        for x, y in tqdm(dl_tr, desc=f"epoch {epoch}/{args.epochs}"):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, enabled=(device == "cuda")):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * len(y)
            total_n += len(y)
        scheduler.step()

        yv, pv = predict(model, dl_va, device)
        val_pred = pv.argmax(1)
        val_acc = accuracy_score(yv, val_pred)
        val_f1 = f1_score(yv, val_pred, average="macro", labels=range(len(CLASSES)), zero_division=0)
        print(f"epoch {epoch:2d}: train_loss={total_loss / max(total_n, 1):.3f} "
              f"val_acc={val_acc:.3f} val_macro_f1={val_f1:.3f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), ckpt)
            print(f"Сохранена лучшая модель: {ckpt}")

    model.load_state_dict(torch.load(ckpt, map_location=device))
    yt, pt = predict(model, dl_te, device)
    pred = pt.argmax(1)
    acc = accuracy_score(yt, pred)
    f1 = f1_score(yt, pred, average="macro", labels=range(len(CLASSES)), zero_division=0)
    print(f"\nИТОГ (CREMA-D test, wav2vec2 {args.mode}): accuracy={acc:.3f} macro-F1={f1:.3f}")
    print(classification_report(yt, pred, labels=range(len(CLASSES)), target_names=CLASSES, zero_division=0))

    # Save test probabilities in manifest order for later fusion/analysis.
    probs_df = test_df[["stem", "actor", "emotion", "split"]].reset_index(drop=True).copy()
    for i, cls in enumerate(CLASSES):
        probs_df[f"p_{cls}"] = pt[:, i]
    probs_df.to_csv(out_dir / f"audio_cremad_w2v_{args.mode}_test_probs.csv", index=False)

    cm = confusion_matrix(yt, pred, labels=range(len(CLASSES)))
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_xlabel("предсказано")
    ax.set_ylabel("истина")
    fig.tight_layout()
    fig.savefig(out_dir / f"audio_cremad_w2v_{args.mode}_cm.png", dpi=150)
    plt.close(fig)

    with open(out_dir / f"audio_cremad_w2v_{args.mode}_metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "dataset": "CREMA-D", "mode": args.mode, "split": "test",
            "classes": CLASSES, "accuracy": acc, "macro_f1": f1,
            "train_actors": sorted(actor_sets["train"]),
            "val_actors": sorted(actor_sets["val"]),
            "test_actors": sorted(actor_sets["test"]),
        }, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
