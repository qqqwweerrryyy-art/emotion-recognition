import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

CLASSES = ["neutral", "happy", "sad", "angry", "fearful", "disgust", "surprised"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--face_mode", default="partial")
    ap.add_argument("--audio_mode", default="partial")
    args = ap.parse_args()

    face = pd.read_csv(f"outputs/face_{args.face_mode}_probs.csv")
    aud = pd.read_csv(f"outputs/audio_{args.audio_mode}_probs.csv")
    pc = [f"p_{c}" for c in CLASSES]
    m = face.merge(aud, on=["video", "actor", "split", "label"], suffixes=("_f", "_a"))
    print(f"Видео: лицо={len(face)}, голос={len(aud)}, совпало={len(m)}")

    pf = m[[c + "_f" for c in pc]].values
    pa = m[[c + "_a" for c in pc]].values
    y = m["label"].values
    is_val = (m["split"] == "val").values
    is_test = (m["split"] == "test").values

    def evaluate(w, mask):
        pred = (w * pf + (1 - w) * pa)[mask].argmax(1)
        return accuracy_score(y[mask], pred), f1_score(y[mask], pred, average="macro")

    grid = np.linspace(0, 1, 11)
    val_f1 = [evaluate(w, is_val)[1] for w in grid]
    best_w = float(grid[int(np.argmax(val_f1))])
    print("Вес лица -> val macro-F1:")
    for w, f in zip(grid, val_f1):
        print(f"  {w:.1f}: {f:.3f}")
    print(f"Лучший вес лица по валидации: {best_w:.1f}\n")

    results = {}
    variants = [("только лицо", 1.0), ("только голос", 0.0),
                ("лицо+голос 50/50", 0.5), (f"лицо+голос (w={best_w:.1f})", best_w)]
    for name, w in variants:
        acc, f1 = evaluate(w, is_test)
        results[name] = {"weight_face": w, "accuracy": acc, "macro_f1": f1}
        print(f"TEST {name:<26} accuracy={acc:.3f}  macro-F1={f1:.3f}")

    pred = (best_w * pf + (1 - best_w) * pa)[is_test].argmax(1)
    print("\nОтчёт по классам (лучший вес):")
    print(classification_report(y[is_test], pred, labels=range(len(CLASSES)),
                                target_names=CLASSES, zero_division=0))

    cm = confusion_matrix(y[is_test], pred, labels=range(len(CLASSES)))
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
    fig.savefig("outputs/fusion_face_audio_cm.png", dpi=150)

    with open("outputs/fusion_face_audio.json", "w", encoding="utf-8") as f:
        json.dump({"best_weight_face": best_w, "test": results}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
