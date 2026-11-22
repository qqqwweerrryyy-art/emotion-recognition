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

# CREMA-D: 6 emotions. "surprised" is not present in this dataset.
CLASSES = ["neutral", "happy", "sad", "angry", "fearful", "disgust"]
CLASS2IDX = {name: i for i, name in enumerate(CLASSES)}


class FaceDS(Dataset):
    def __init__(self, df, tf):
        self.paths = df["path"].astype(str).tolist()
        self.labels = [CLASS2IDX[str(x).strip().lower()] for x in df["emotion"]]
        self.tf = tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            img = im.convert("RGB")
        return self.tf(img), self.labels[i]


def build_model(mode, n_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, n_classes))
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True
    if mode == "partial":
        for p in model.layer4.parameters():
            p.requires_grad = True
    return model


def set_train(model, mode):
    model.eval()  # keep frozen BatchNorm layers in eval mode
    model.fc.train()
    if mode == "partial":
        model.layer4.train()


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device, enabled=(device == "cuda")):
            out = model(x)
        ys.extend(y.tolist())
        ps.extend(out.argmax(1).cpu().tolist())
    return np.asarray(ys), np.asarray(ps)


def main():
    ap = argparse.ArgumentParser(description="Train CREMA-D face emotion model")
    ap.add_argument("--labels", default="data/cremad_processed/labels.csv")
    ap.add_argument("--mode", choices=["frozen", "partial"], default="frozen")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Устройство: {device}")

    labels_path = Path(args.labels)
    if not labels_path.exists():
        raise SystemExit(f"Не найден файл разметки: {labels_path}")
    df = pd.read_csv(labels_path)

    required = {"path", "actor", "emotion", "split"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"В labels.csv не хватает столбцов: {sorted(missing)}")

    df = df.dropna(subset=["path", "actor", "emotion", "split"]).copy()
    df["emotion"] = df["emotion"].astype(str).str.strip().str.lower()
    df["split"] = df["split"].astype(str).str.strip().str.lower()
    unknown = sorted(set(df["emotion"]) - set(CLASSES))
    if unknown:
        raise SystemExit(f"Обнаружены неизвестные эмоции: {unknown}")
    df = df[df["split"].isin(["train", "val", "test"])].copy()

    for split in ["train", "val", "test"]:
        count = int((df["split"] == split).sum())
        print(f"Кадров {split}: {count}")
        if count == 0:
            raise SystemExit(f"В labels.csv нет примеров split={split}. Сначала проверь подготовку CREMA-D.")

    actor_sets = {s: set(df.loc[df["split"] == s, "actor"].astype(str)) for s in ["train", "val", "test"]}
    if actor_sets["train"] & actor_sets["val"] or actor_sets["train"] & actor_sets["test"] or actor_sets["val"] & actor_sets["test"]:
        raise SystemExit("ОШИБКА: один и тот же актёр попал в несколько split!")
    print("Актёры по split:", {s: len(v) for s, v in actor_sets.items()})
    print("Классы:", CLASSES)

    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    tf_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        norm,
    ])
    tf_eval = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), norm])

    def make_loader(split, tf, shuffle):
        subset = df[df["split"] == split].reset_index(drop=True)
        ds = FaceDS(subset, tf)
        return DataLoader(
            ds, batch_size=args.batch, shuffle=shuffle,
            num_workers=args.workers, pin_memory=(device == "cuda")
        )

    dl_tr = make_loader("train", tf_train, True)
    dl_va = make_loader("val", tf_eval, False)
    dl_te = make_loader("test", tf_eval, False)

    model = build_model(args.mode, len(CLASSES)).to(device)
    groups = [{"params": model.fc.parameters(), "lr": args.lr_head}]
    if args.mode == "partial":
        groups.append({"params": model.layer4.parameters(), "lr": 1e-4})
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler(device, enabled=(device == "cuda"))
    criterion = nn.CrossEntropyLoss()

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    ckpt = out_dir / f"face_cremad_{args.mode}.pth"
    best_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        set_train(model, args.mode)
        total_loss, total_n = 0.0, 0
        for x, y in tqdm(dl_tr, desc=f"epoch {epoch}/{args.epochs}"):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, enabled=(device == "cuda")):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total_loss += loss.item() * len(y)
            total_n += len(y)

        sched.step()
        yv, pv = predict(model, dl_va, device)
        val_acc = accuracy_score(yv, pv)
        val_f1 = f1_score(yv, pv, average="macro", labels=range(len(CLASSES)), zero_division=0)
        print(f"epoch {epoch:2d}: train_loss={total_loss / max(total_n, 1):.3f} "
              f"val_acc={val_acc:.3f} val_macro_f1={val_f1:.3f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), ckpt)
            print(f"Сохранена лучшая модель: {ckpt}")

    model.load_state_dict(torch.load(ckpt, map_location=device))
    yt, pt = predict(model, dl_te, device)
    acc = accuracy_score(yt, pt)
    f1 = f1_score(yt, pt, average="macro", labels=range(len(CLASSES)), zero_division=0)
    print(f"\nИТОГ (CREMA-D test, face {args.mode}): accuracy={acc:.3f} macro-F1={f1:.3f}")
    print(classification_report(yt, pt, labels=range(len(CLASSES)), target_names=CLASSES, zero_division=0))

    cm = confusion_matrix(yt, pt, labels=range(len(CLASSES)))
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
    fig.savefig(out_dir / f"face_cremad_{args.mode}_cm.png", dpi=150)
    plt.close(fig)

    with open(out_dir / f"face_cremad_{args.mode}_metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "dataset": "CREMA-D", "mode": args.mode, "split": "test",
            "classes": CLASSES, "accuracy": acc, "macro_f1": f1,
            "train_actors": sorted(actor_sets["train"]),
            "val_actors": sorted(actor_sets["val"]),
            "test_actors": sorted(actor_sets["test"]),
        }, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
