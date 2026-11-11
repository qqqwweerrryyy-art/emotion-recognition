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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

CLASSES = ["neutral", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
CODE2CLS = {1: 0, 2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6}  # calm -> neutral


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
           "-vn", "-ac", "1", "-ar", "16000", str(out)]
    subprocess.run(cmd, check=True)


def features(path):
    y, sr = librosa.load(path, sr=16000)
    y, _ = librosa.effects.trim(y, top_db=30)
    if len(y) < 2048:
        y = np.pad(y, (0, 2048 - len(y)))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    d = librosa.feature.delta(mfcc)
    rms = librosa.feature.rms(y=y)
    zcr = librosa.feature.zero_crossing_rate(y)
    parts = [mfcc, d, rms, zcr]
    return np.concatenate([np.r_[p.mean(1), p.std(1)] for p in parts])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/ravdess")
    ap.add_argument("--audio", default="data/audio")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg не найден в PATH")

    videos = sorted(Path(args.data).rglob("01-*.mp4"))
    print(f"Найдено видео: {len(videos)}")
    audio_dir = Path(args.audio)
    audio_dir.mkdir(parents=True, exist_ok=True)

    X, y, actors = [], [], []
    for v in tqdm(videos, desc="звук и признаки"):
        wav = audio_dir / f"{v.stem}.wav"
        extract_wav(v, wav)
        parts = v.stem.split("-")
        X.append(features(wav))
        y.append(CODE2CLS[int(parts[2])])
        actors.append(int(parts[6]))
    X, y, actors = np.array(X), np.array(y), np.array(actors)

    tr_a, va_a, te_a = split_actors(sorted(set(actors.tolist())), args.seed)
    print(f"Актёры: train={tr_a} val={va_a} test={te_a}")
    m_tr, m_va, m_te = (np.isin(actors, a) for a in (tr_a, va_a, te_a))

    best = (-1, None, None)
    for C in [0.1, 1, 10, 100]:
        clf = make_pipeline(StandardScaler(),
                            SVC(kernel="rbf", C=C, class_weight="balanced", probability=True,
                                random_state=args.seed))
        clf.fit(X[m_tr], y[m_tr])
        f1 = f1_score(y[m_va], clf.predict(X[m_va]), average="macro")
        print(f"C={C:<6} val_macro_F1={f1:.3f}")
        if f1 > best[0]:
            best = (f1, C, clf)
    _, C, clf = best
    name, mask = ("test", m_te) if m_te.any() else ("val", m_va)
    pred = clf.predict(X[mask])
    acc, f1 = accuracy_score(y[mask], pred), f1_score(y[mask], pred, average="macro")
    print(f"\nИТОГ ({name}, SVM, C={C}): accuracy={acc:.3f}  macro-F1={f1:.3f}")
    print(classification_report(y[mask], pred, labels=range(len(CLASSES)),
                                target_names=CLASSES, zero_division=0))

    Path("outputs").mkdir(exist_ok=True)
    cm = confusion_matrix(y[mask], pred, labels=range(len(CLASSES)))
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
    fig.savefig("outputs/audio_svm_cm.png", dpi=150)
    with open("outputs/audio_svm_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"model": "svm_mfcc", "split": name, "C": C, "accuracy": acc, "macro_f1": f1,
                   "train_actors": tr_a, "val_actors": va_a, "test_actors": te_a},
                  f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
