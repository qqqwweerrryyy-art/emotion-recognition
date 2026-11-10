import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

EMOTIONS = {1: "neutral", 2: "calm", 3: "happy", 4: "sad",
            5: "angry", 6: "fearful", 7: "disgust", 8: "surprised"}


def load_detector():
    path = Path("models/haarcascade_frontalface_default.xml")
    det = cv2.CascadeClassifier(str(path))
    if det.empty():
        raise RuntimeError(f"Не удалось загрузить детектор лиц: {path}")
    return det


def crop_face(det, frame, margin=0.3, size=224):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = det.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])  # самое крупное лицо
    cx, cy = x + w // 2, y + h // 2
    half = int(max(w, h) * (1 + 2 * margin) / 2)
    x0, y0 = max(cx - half, 0), max(cy - half, 0)
    x1, y1 = min(cx + half, frame.shape[1]), min(cy + half, frame.shape[0])
    return cv2.resize(frame[y0:y1, x0:x1], (size, size))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/ravdess")
    ap.add_argument("--out", default="data/faces")
    ap.add_argument("--frames", type=int, default=10, help="кадров на видео")
    ap.add_argument("--limit", type=int, default=0, help="обработать только N видео (для теста)")
    args = ap.parse_args()

    videos = sorted(Path(args.data).rglob("01-*.mp4"))
    if args.limit:
        videos = videos[: args.limit]
    print(f"Найдено видео: {len(videos)}")
    if not videos:
        return

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    det = load_detector()
    rows, no_face, saved = [], 0, 0

    for v in tqdm(videos):
        parts = v.stem.split("-")
        emo, actor = int(parts[2]), int(parts[6])
        cap = cv2.VideoCapture(str(v))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            print("Не удалось прочитать:", v)
            continue
        idxs = np.linspace(0.2 * total, 0.8 * total, args.frames).astype(int)
        actor_dir = out / f"actor_{actor:02d}"
        actor_dir.mkdir(exist_ok=True)
        for k, i in enumerate(idxs):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, frame = cap.read()
            if not ok:
                continue
            face = crop_face(det, frame)
            if face is None:
                no_face += 1
                continue
            p = actor_dir / f"{v.stem}_f{k}.jpg"
            cv2.imwrite(str(p), face)
            rows.append([p.as_posix(), v.stem, actor, emo, EMOTIONS[emo]])
            saved += 1
        cap.release()

    with open(out / "labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "video", "actor", "emotion_code", "emotion"])
        w.writerows(rows)
    print(f"Сохранено лиц: {saved}, кадров без лица: {no_face}")


if __name__ == "__main__":
    main()

