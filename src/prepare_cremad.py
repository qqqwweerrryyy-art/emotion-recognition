import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


# Совместимость с CODE2CLS в src/train_face.py и src/train_audio.py:
# 1=neutral, 3=happy, 4=sad, 5=angry, 6=fearful, 7=disgust.
CREMA_EMOTIONS = {
    "NEU": (1, "neutral"),
    "HAP": (3, "happy"),
    "SAD": (4, "sad"),
    "ANG": (5, "angry"),
    "FEA": (6, "fearful"),
    "DIS": (7, "disgust"),
}


def parse_stem(path):
    """CREMA-D: Actor_Sentence_Emotion_Intensity.wav/flv"""
    parts = path.stem.split("_")
    if len(parts) != 4:
        raise ValueError(f"Неожиданный формат имени файла: {path.name}")
    actor, sentence, emotion, intensity = parts
    if emotion not in CREMA_EMOTIONS:
        raise ValueError(f"Неизвестный код эмоции в {path.name}: {emotion}")
    return int(actor), sentence, emotion, intensity


def make_actor_split(actors, seed):
    actors = sorted(set(actors))
    if len(actors) != 91:
        raise RuntimeError(
            f"Ожидалось 91 уникальный актёр CREMA-D, найдено {len(actors)}. "
            "Проверьте полноту AudioWAV."
        )
    shuffled = actors.copy()
    random.Random(seed).shuffle(shuffled)
    test = sorted(shuffled[:15])
    val = sorted(shuffled[15:23])
    train = sorted(shuffled[23:])
    assert len(train) == 68 and len(val) == 8 and len(test) == 15
    assert not (set(train) & set(val) or set(train) & set(test) or set(val) & set(test))
    return {
        "seed": seed,
        "train": train,
        "val": val,
        "test": test,
    }


def load_detector(cascade_path):
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise RuntimeError(f"Не удалось загрузить каскад: {cascade_path}")
    return detector


def crop_face(detector, frame, margin=0.3, size=224):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    cx, cy = x + w // 2, y + h // 2
    half = int(max(w, h) * (1 + 2 * margin) / 2)
    x0, y0 = max(cx - half, 0), max(cy - half, 0)
    x1, y1 = min(cx + half, frame.shape[1]), min(cy + half, frame.shape[0])
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (size, size))


def main():
    parser = argparse.ArgumentParser(
        description="Подготовка CREMA-D: аудиоманифест, split актёров и кадры лиц."
    )
    parser.add_argument("--data", default="data/cremad")
    parser.add_argument("--out", default="data/cremad_processed")
    parser.add_argument("--frames", type=int, default=10,
                        help="число равномерно выбранных кадров на видео")
    parser.add_argument("--limit", type=int, default=0,
                        help="обработать не более N видео (только для пробного запуска)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sentences", default="",
                        help="необязательный фильтр, например IEO,TIE,DFA")
    parser.add_argument("--no_faces", action="store_true",
                        help="создать манифесты, не извлекая кадры")
    parser.add_argument(
        "--cascade",
        default="models/haarcascade_frontalface_default.xml"
    )
    args = parser.parse_args()

    root = Path(args.data)
    out = Path(args.out)
    audio_dir = root / "AudioWAV"
    video_dir = root / "VideoFlash"
    if not audio_dir.is_dir():
        raise SystemExit(f"Не найдена папка: {audio_dir}")
    if not video_dir.is_dir():
        raise SystemExit(f"Не найдена папка: {video_dir}")
    if args.frames < 1:
        raise SystemExit("--frames должен быть не меньше 1")

    audio_files = sorted(audio_dir.glob("*.wav"))
    if not audio_files:
        raise SystemExit(f"В {audio_dir} не найдены WAV-файлы")

    records = []
    for wav in audio_files:
        actor, sentence, emotion_code, intensity = parse_stem(wav)
        records.append({
            "stem": wav.stem,
            "audio_path": wav.as_posix(),
            "actor": actor,
            "sentence": sentence,
            "emotion_token": emotion_code,
            "emotion_code": CREMA_EMOTIONS[emotion_code][0],
            "emotion": CREMA_EMOTIONS[emotion_code][1],
            "intensity": intensity,
        })

    split = make_actor_split([r["actor"] for r in records], args.seed)
    actor_to_split = {
        actor: name
        for name in ("train", "val", "test")
        for actor in split[name]
    }
    for record in records:
        record["split"] = actor_to_split[record["actor"]]

    out.mkdir(parents=True, exist_ok=True)
    split_path = out / f"actor_split_seed{args.seed}.json"
    # Сохраняем фактический seed в имени содержимого, а не полагаемся на имя файла.
    split_path.write_text(
        json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    manifest_path = out / "audio_manifest.csv"
    audio_columns = [
        "stem", "audio_path", "actor", "sentence", "emotion_token",
        "emotion_code", "emotion", "intensity", "split"
    ]
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=audio_columns)
        writer.writeheader()
        writer.writerows(records)

    print(f"Найдено аудио: {len(records)}")
    print(f"Актёры: train={len(split['train'])}, val={len(split['val'])}, "
          f"test={len(split['test'])}")
    print(f"Манифест аудио: {manifest_path}")
    print(f"Разделение актёров: {split_path}")
    print("Количество аудиозаписей по split:",
          dict(Counter(r["split"] for r in records)))

    if args.no_faces:
        print("Извлечение лиц пропущено (--no_faces).")
        return

    videos = []
    sentence_filter = {
        item.strip().upper()
        for item in args.sentences.split(",") if item.strip()
    }
    for record in records:
        if sentence_filter and record["sentence"].upper() not in sentence_filter:
            continue
        video = video_dir / f"{record['stem']}.flv"
        if video.is_file():
            videos.append((record, video))
    videos.sort(key=lambda item: item[1].name)

    missing_videos = len(records) - sum(
        1 for r in records if (video_dir / f"{r['stem']}.flv").is_file()
    )
    if missing_videos:
        print(f"Внимание: не найдены соответствующие FLV: {missing_videos}")
    if args.limit:
        videos = videos[:args.limit]
    print(f"Видео для извлечения лиц: {len(videos)}")

    detector = load_detector(Path(args.cascade))
    faces_root = out / "faces"
    faces_root.mkdir(parents=True, exist_ok=True)
    labels_path = out / "labels.csv"
    rows = []
    no_face_frames = 0
    unreadable = 0

    for record, video_path in tqdm(videos, desc="Извлечение лиц"):
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            unreadable += 1
            capture.release()
            continue
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            unreadable += 1
            capture.release()
            continue

        # Берём кадры из центральных 60% ролика, как в существующем extract_faces.py.
        indexes = np.linspace(
            0.2 * total_frames, 0.8 * total_frames, args.frames
        ).astype(int)
        actor_dir = faces_root / f"actor_{record['actor']:04d}"
        actor_dir.mkdir(parents=True, exist_ok=True)

        for frame_number, frame_index in enumerate(indexes):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = capture.read()
            if not ok:
                continue
            face = crop_face(detector, frame)
            if face is None:
                no_face_frames += 1
                continue
            image_path = actor_dir / f"{record['stem']}_f{frame_number:02d}.jpg"
            if not cv2.imwrite(str(image_path), face):
                continue
            rows.append({
                "path": image_path.as_posix(),
                "video": record["stem"],
                "actor": record["actor"],
                "emotion_code": record["emotion_code"],
                "emotion": record["emotion"],
                "split": record["split"],
            })
        capture.release()

    label_columns = ["path", "video", "actor", "emotion_code", "emotion", "split"]
    with labels_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=label_columns)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Сохранено кадров с лицами: {len(rows)}")
    print(f"Кадров без обнаруженного лица: {no_face_frames}")
    print(f"Нечитаемых видео: {unreadable}")
    print(f"Таблица лиц: {labels_path}")
    print("Кадры по split:", dict(Counter(r["split"] for r in rows)))


if __name__ == "__main__":
    main()
