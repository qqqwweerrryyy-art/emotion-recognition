import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

CLASSES = ["neutral", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
CODE2CLS = {1: 0, 2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6}  # calm -> neutral


class FaceDS(Dataset):
    def __init__(self, df, tf):
        self.paths = df["path"].tolist()
        self.labels = [CODE2CLS[c] for c in df["emotion_code"]]
        self.tf = tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.tf(img), self.labels[i]


def split_actors(actors, seed):
    actors = sorted(actors)
    random.Random(seed).shuffle(actors)
    n = len(actors)
    n_test = round(0.2 * n) if n >= 6 else 0
    n_val = max(1, round(0.2 * n))
    test = actors[:n_test]
    val = actors[n_test:n_test + n_val]
    train = actors[n_test + n_val:]
    return train, val, test


def build_model(mode, n_classes):
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, n_classes))
    for p in m.parameters():
        p.requires_grad = False
    for p in m.fc.parameters():
        p.requires_grad = True
    if mode == "partial":
        for p in m.layer4.parameters():
            p.requires_grad = True
    return m


def set_train(model, mode):
    model.eval()  # замороженные части (включая BatchNorm) остаются в eval
    model.fc.train()
    if mode == "partial":
        model.layer4.train()


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device)
        with torch.autocast(device_type=device, enabled=(device == "cuda")):
            out = model(x)
        ps.extend(out.argmax(1).cpu().tolist())
        ys.extend(y.tolist())
    return np.array(ys), np.array(ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/faces/labels.csv")
    ap.add_argument("--mode", choices=["frozen", "partial"], default="frozen")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(args.labels)
    tr_a, va_a, te_a = split_actors(df["actor"].unique().tolist(), args.seed)
    print(f"Актёры: train={tr_a} val={va_a} test={te_a}")
    if len(df["actor"].unique()) < 12:
        print("ВНИМАНИЕ: актёров мало, результат будет шумным (это тестовый прогон).")

    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    tf_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(), norm])
    tf_eval = transforms.Compose([transforms.ToTensor(), norm])

    def loader(actors, tf, shuffle):
        sub = df[df["actor"].isin(actors)]
        return DataLoader(FaceDS(sub, tf), batch_size=args.batch, shuffle=shuffle,
                          num_workers=2, pin_memory=(device == "cuda"))

    dl_tr = loader(tr_a, tf_train, True)
    dl_va = loader(va_a, tf_eval, False)
    dl_te = loader(te_a, tf_eval, False) if te_a else None

    model = build_model(args.mode, len(CLASSES)).to(device)
    groups = [{"params": model.fc.parameters(), "lr": args.lr_head}]
    if args.mode == "partial":
        groups.append({"params": model.layer4.parameters(), "lr": 1e-4})
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler(device, enabled=(device == "cuda"))
    crit = nn.CrossEntropyLoss()

    Path("outputs").mkdir(exist_ok=True)
    ckpt = Path(f"outputs/face_{args.mode}.pth")
    best_f1 = -1.0

    for ep in range(1, args.epochs + 1):
        set_train(model, args.mode)
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
        acc, f1 = accuracy_score(yv, pv), f1_score(yv, pv, average="macro")
        print(f"epoch {ep:2d}: train_loss={total / n:.3f}  val_acc={acc:.3f}  val_f1={f1:.3f}")
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), ckpt)

    model.load_state_dict(torch.load(ckpt, map_location=device))
    name, dl_final = ("test", dl_te) if dl_te else ("val", dl_va)
    y, p = predict(model, dl_final, device)
    acc, f1 = accuracy_score(y, p), f1_score(y, p, average="macro")
    print(f"\nИТОГ ({name}, режим {args.mode}): accuracy={acc:.3f}  macro-F1={f1:.3f}")
    print(classification_report(y, p, labels=range(len(CLASSES)), target_names=CLASSES, zero_division=0))

    cm = confusion_matrix(y, p, labels=range(len(CLASSES)))
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
    fig.savefig(f"outputs/face_{args.mode}_cm.png", dpi=150)

    with open(f"outputs/face_{args.mode}_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"mode": args.mode, "split": name, "accuracy": acc, "macro_f1": f1,
                   "train_actors": tr_a, "val_actors": va_a, "test_actors": te_a}, f,
                  ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

